# The alchemical topology plan (A1, callable layer)

**Under construction, not in any release.** This describes the callable layer on
`work/0.7.0-topology`. There is no `md-openmm combine-topology` command yet; S0 wires the CLI.

## What it is

```python
from md_tools.alchemy.topology import Environment, build_topology_plan, load_plan
from md_tools.alchemy.topology_mapping import AtomMap

plan = build_topology_plan(package_a, package_b, AtomMap.from_pairs(package_a, package_b, pairs),
                           Environment.from_files("built.xml", "built.pdb", selector),
                           mode="hybrid")          # or "single", "dual"
plan.write("plan/")                                # plan.json, system_{a,b}.xml, combined.pdb,
                                                   # positions.npy -- a new directory, never over
plan = load_plan("plan/", package_roots=[catalog]) # every digest re-verified
```

`plan.system_a` and `plan.system_b` are the two endpoint Systems; `plan.common`, `plan.a_only`
and `plan.b_only` are the particle sets (frozensets of plan indices); `plan.record` is the JSON
record, schema `md-tools-topology-plan/1`, and `plan.sha256` its digest.

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

- **Dummies** have charge 0 and epsilon 0 (sigma kept) and every exception touching them is zero:
  no nonbonded interaction with anything, themselves included.
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

## Constraints and masses

The environment's constraint policy is read from its ligand (HBonds, AllBonds or None) and
applied to endpoint B. A constraint that would appear or vanish along the path, or change length,
is refused -- a constraint has one length. An environment whose ligand has only X-H bonds cannot
say whether it was built with HBonds or AllBonds; it is refused when endpoint B has a heavy-atom
bond (untested: no fixture package has only X-H bonds).

Masses are the environment's for every existing particle (endpoint A's for the core) and package
B's for appended ones; core mass changes are recorded. Masses do not enter the configurational
free energy. An environment with repartitioned hydrogen masses is refused: appended atoms would
need the same repartitioning, which is not implemented.

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

  with E_dummy and E_restraint computed in numpy from the record and dE_dispersion the change in
  OpenMM's dispersion correction. **That correction averages over every particle, zero-epsilon
  dummies included**, so a dummy shifts it (-1.2e-4 kJ/mol for one dummy in the fixture box). It
  is a real term, not noise, and a Hamiltonian that moves LJ out of `NonbondedForce` changes it.

The tests build the reference endpoint from scratch with `ForceField("amber14/tip3p.xml")` plus
the package's ffxml, and hold every force class to 1e-7 kJ/mol on the Reference platform, after
asserting the raw difference is large enough that the check cannot pass by accident.

## Proposed `combine-topology` input (PROPOSED -- S0 owns the final schema)

A YAML file, unknown keys refused, paths relative to the file:

```yaml
format: md-tools-combine-topology/1     # proposed
mode: hybrid                            # single | hybrid | dual; `separated` is refused by name
endpoints:
  A:
    parameters: LOCAL-OTMSDBZUPAUEDD/param_cf41a2bd76f4   # catalog reference, or a package path,
  B:                                                       # resolved by ligands.catalog.resolve_package
    parameters: ./chloroethane/parameter
environment:                            # ONE environment, holding endpoint A; B never has its own
  system: build/built.xml
  topology: build/built.pdb
  ligand: {resname: ETA}                # a LigandSelector naming exactly one residue:
                                        # {resname} or {chain, resid, insertion_code}
map:
  file: ethane-chloroethane.map.yaml    # explicit pairs, or a stored map record (below)
  # automatic: {...}                    # NOT IMPLEMENTED; reserved for a proposal checked by
                                        # validate_map and written out for review
dual:
  restraint_k_kj_mol_nm2: 1000.0        # dual mode only; refused in the other modes
b_pose: null                            # optional .sdf with B's pose in B package order; default:
                                        # superpose B's reference conformer on the mapped A atoms
output: plan/                           # a NEW directory; an existing one is refused
```

The map file is either explicit pairs, by package atom name or index,

```yaml
pairs:
  - [C1, C1]
  - [C2, C2]
  - [H1, H1]
```

or a map record as `AtomMap.record` writes it (schema `md-tools-alchemical-atom-map/1`, both
directions and a digest), which `AtomMap.from_record` re-verifies against the two packages.

The command maps onto the callable layer as: resolve both packages, `Environment.from_files`,
`AtomMap.from_pairs` / `from_record`, `build_topology_plan(..., mode=...)`, `plan.write(output)`.
Every refusal happens in `build_topology_plan`, before anything is written.

## Fixtures

`tests/data/alchemy/v1/` (see its README): ethane, chloroethane and ethanol packages (AM1-BCC,
openff-2.2.1) and ethane in TIP3P built by `build-top`. Loaders in `tests/alchemy_fixtures.py`.
