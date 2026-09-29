# Alchemical topology

Two registered ligand parameter packages, one built environment holding the first of them, and an
atom map between them, combined into **one topology plan**: two endpoint Systems over a single
particle index space, plus the record that says exactly how they were built.

```bash
md-openmm combine-topology --config example.config -odir ./plan
```

## What this is today, and what it is not

`combine-topology` is under construction for 0.7.0. The honest statement, stated precisely because
the halfway point is where a reader is most likely to assume the rest:

* **A plan is a finished artefact.** The command writes one, and
  `md_tools.alchemy.topology.load_plan` reads it back with every digest re-verified.
* **`protocol: alchemical` resolves.** `build-md` accepts an `alchemical` section naming a plan, a
  λ path, a window placement, the Amber18 softcore settings and the per-window lengths, and
  `md-run` parses the matching `&alchemical` block. Every refusal that belongs to one of those
  values fires where it is written.
* **Neither surface runs a window yet.** `build-md` refuses to generate the run directory and
  `md-run` refuses to dispatch, each by name. Left to fall through, the first would write scripts
  that minimise, equilibrate and stop — a complete-looking run that sampled nothing — and the
  second would report the success of a campaign it never ran.
* **The window runtime itself works**, through the Python API
  (`md_tools.alchemy.windows.run_window`, `md_tools.alchemy.campaign`), and has produced absolute
  hydration free energies that agree with an independent alchemical route — see
  [validation.md](validation.md).

So a campaign today is driven from Python, not from a generated directory. What is missing is the
step between the configuration and the runtime, and both surfaces say so rather than approximating
it.

## The example file beside this README

| file | what it is |
|---|---|
| [`example.config`](example.config) | what you hand to `md-openmm combine-topology` |

There is no `example.in` here: `build-md` does not yet write one for an alchemical ladder, and a
hand-written example of a file no command produces is a file nothing checks.

## The command

| flag | meaning |
|---|---|
| `--config PATH` | the configuration, format `md-tools-combine-topology/1` |
| `-odir DIR` | where the plan is written. **It must not exist**: a plan is written once, into a new directory, staged and renamed into place |
| `--check` | build the plan, run every check, and write **nothing** — not even `DIR`, not even its parent |

A configuration says *what* to combine; `-odir` says where it goes, as it does for `build-md`.
Exit status is 0 on success, 2 on a refusal that names its cause, 1 on anything else.

## What a plan directory holds

| file | what it is |
|---|---|
| `plan.json` | the record: mode, junction policy, the atom map and its digest, both endpoints' package references and parameter digests, the environment's identity and solvation, the particle numbering, every check that ran and what it found, the SHA-256 of the other four files, and `plan_sha256` over all of it |
| `system_a.xml` | endpoint A's serialised OpenMM System — the environment System, unchanged, plus the appended endpoint-B particles and terms |
| `system_b.xml` | endpoint B's System: the same particles, masses, constraints and force layout, term slot for term slot; only parameter values differ |
| `combined.pdb` | the combined topology, written from the exact bytes the plan was built against |
| `positions.npy` | the combined coordinates in nm, little-endian float64, no pickle |

Every environment particle — solvent, ions, protein and the endpoint-A ligand — keeps the index it
had in the environment System. The particles only endpoint B has are **appended**, as one new
residue in a new chain. Nothing is renumbered, so an atom index or a one-based residue index
resolved against the environment before combination still means the same atom afterwards.

A map proposed with `automatic: true` is written **beside** the plan, as `<odir>.map.yaml`, not
inside it: a plan directory holds exactly the five files its record names. That sidecar, handed
back as `map: {file: ...}`, reproduces the same plan — the same `plan_sha256` — which is what makes
"review the proposal, then commit to it" a checkable step rather than a hope. An existing sidecar
is never overwritten either.

## The configuration, key by key

Every file is YAML despite the `.config` suffix, and an unknown key is refused by name.

| key | type | default | meaning |
|---|---|---|---|
| `format` | string | required | must be `md-tools-combine-topology/1` |
| `mode` | string | required | `single`, `hybrid` or `dual`. `separated` is refused by name |
| `b_pose` | path | `null` | an `.sdf` holding endpoint B's pose, relative to the configuration |
| `endpoints.A` | mapping | required | `{parameters: <compound>/param_<id>}` — the endpoint the environment already holds |
| `endpoints.B` | mapping | required | `{parameters: ...}` — the endpoint it becomes |
| `environment.system` | path | required | the built System holding endpoint A, e.g. `build/built.xml` |
| `environment.topology` | path | required | its topology, e.g. `build/built.pdb` |
| `environment.record` | path | required | the `build-top` record of that build, e.g. `build/built.log` |
| `environment.ligand` | mapping | required | a selector naming exactly ONE residue: `{resname}` or `{chain, resid, insertion_code}` |
| `map.file` | path | `null` | explicit pairs, or a stored atom-map record |
| `map.automatic` | bool | `false` | propose the map (`rdkit-fmcs-heavy/1`) |
| `dual.restraint_k_kj_mol_nm2` | number ≥ 0 | `null` (builder's default, 1000) | the restraint holding the two dual-topology ligands together; `mode: dual` only |
| `ligand_catalog.path` | path | `null` | a catalog searched FIRST for `<compound>/param_<id>` references, before `$MD_DATA/parameters/ligands` |

`endpoints.A` and `endpoints.B` accept **exactly one key each**, `parameters`; anything else is
refused by name rather than ignored. `ligand_catalog.path` stays in the configuration as written
and is resolved against the configuration file when the catalog is actually searched, so no record
carries this machine's absolute path.

The packages' parameters are used **exactly as registered**. Nothing is reparameterised, and an
environment whose ligand does not carry package A's parameters is refused rather than adjusted.

`b_pose` must be endpoint B's chemical state: its atoms are put into package order by the ligand
module's own matcher, which refuses a different molecule rather than guessing a correspondence.
Left unset, package B's reference conformer is superposed on the mapped A atoms (Kabsch), and the
core RMSD of that fit is recorded in the plan.

## The three modes

```text
single   ONE evolving atom representation: every atom of one endpoint is mapped, and a mapped
         atom may change element -- a hydrogen may become a heavy atom where no constraint
         forbids it.
hybrid   a mapped core plus endpoint-unique atoms on BOTH sides, as separate particles. A
         hydrogen is never mapped to a heavy atom.
dual     the two ligands share no particle. Each is a complete molecule, B appended whole; every
         A-B pair is an explicit zero exception; a harmonic restraint between the centroids of
         the mapped atom groups keeps the dummy ligand with the physical one.
```

The mode is not a preference, it is a claim about the map, and each mode checks the claim:

* `single` with any unmapped atom on both sides is refused — that is a hybrid topology, and the
  refusal says so with the counts.
* `hybrid` (and `single`) require the mapped core to keep its bond graph in **both** directions,
  and every mapped atom to be in the same number of rings at both endpoints.
* `dual` uses the map only to place endpoint B and to define the restrained centroid groups; the
  two ligands never share a particle, so the core-preservation checks do not apply to it.

`dual.restraint_k_kj_mol_nm2` set under any other mode is refused, because no other mode has that
restraint and a setting that is accepted and inert is worse than one refused.

`separated` topology is deferred beyond 0.7.0 and is **not partially supported**: it is refused by
name, before anything else in the configuration is resolved.

## How a map is supplied

Exactly one of `map.file` and `map.automatic: true`. Neither, or both, is refused.

A map is stated in **package-local** atom indices or names, of the two registered packages. Those
are the only atom identities that survive a rebuild: a package's atom *i* is the *i*-th atom of its
`molecule.sdf` forever, while a System index moves the moment solvent, ions or a protein are added.
A name used in a map must name exactly one atom of its package.

`map.file` reads either form:

```yaml
pairs: [[C1, C1], [C2, C2], [H11, H11]]     # by package atom name, or by index
```

or an atom-map record as `combine-topology` itself writes it — schema
`md-tools-alchemical-atom-map/1`, carrying both directions by index and by name plus a digest. A
stored record is re-verified in full: if `a_to_b` and `b_to_a` are not exact inverses, or the
record does not hash back to itself, it is refused as edited or as belonging to other packages.
That is why hand-editing a proposal and expecting it to be accepted does not work; state the edited
map as `pairs:` instead.

`map.automatic: true` proposes one. The method is `rdkit-fmcs-heavy/1`: a maximum common
substructure over heavy atoms with elements compared exactly, bond orders compared exactly, rings
matching only rings and only complete rings, with a 10 s timeout; every placement of that
substructure in A and in B is enumerated; the hydrogens of each mapped heavy pair are paired by
distance after superposing B's conformer on A's mapped heavy atoms. Each candidate must pass every
chemistry check below, and the largest survivors win.

An automatic map refuses when:

* **the mode is `single`.** A single-topology map decides which atoms *become* which, and a mapped
  atom may change element — that is stated, not proposed.
* **the search did not finish** within its 10 s timeout.
* **the two molecules share no heavy-atom substructure.**
* **no placement validates.** The refusal lists the first refusals it collected, so you can see
  which chemistry rule stopped every candidate.
* **the proposal is chemically ambiguous**: two or more maps of the same, largest size validate and
  are *not* related by a symmetry of either molecule (compared by canonical ranks with ties
  unbroken). The choice would then change the transformation, and nothing in the proposer can
  make it. The alternatives are listed and an explicit map is required.

That last one is the reason the proposer exists in this shape at all. Maps related by a symmetry of
one of the molecules describe the same transformation, so picking among them is safe; maps that are
not related by a symmetry describe *different* transformations that would both run and both produce
a number.

## The chemistry refusals, and why each exists

These run in `md_tools.alchemy.topology_mapping.validate_map`, before anything is built. Each is a
transformation the 0.7.0 construction does not represent correctly — and a plan built from one would
run and give a number, which is the whole hazard.

| refused | why |
|---|---|
| **net formal charge changes** | a charge-changing transformation needs a finite-size / co-alchemical treatment 0.7.0 does not implement; it is refused rather than run uncorrected |
| **the two packages come from different force fields** | the two endpoints of one transformation must come from ONE force field |
| **the charge `method` or `backend_id` differs** | two charge models in one transformation measure the model difference along with the chemistry |
| **the nonbonded conventions differ** | the 1-4 scales the two packages were built under must agree |
| **the core's bond graph changes** | a mapped A bond landing on two unbonded B atoms (or the reverse) is a ring opening, a ring closure or a rearrangement |
| **ring membership changes** | a mapped atom in a different number of rings at the two endpoints is a ring change, refused in 0.7.0 |
| **a unique group attached by more than one bond** | a dummy group's partition function separates from the physical system only when it hangs from ONE bond to ONE core atom (the single-anchor rule of Fleck, Wieder and Boresch, *J. Chem. Theory Comput.* 2021, **17**, 4403). A group bridging two core atoms — a ring growing, or a partially mapped ring — is refused |
| **a stereocentre with fewer than three mapped neighbours in common** | with fewer than three, the map does not *say* which configuration B has relative to A |
| **an inverted stereocentre** | checked by the signed volume of three shared neighbours in the two package conformers; a sign flip is a stereochemistry change |
| **a hydrogen mapped to a heavy atom in `hybrid` or `dual`** | that is a single-topology construction. In hybrid topology, leave both unmapped as unique atoms |
| **a map that is not one-to-one, or empty** | an atom mapped twice on either side, or no pairs at all |

Element changes, bond-order changes and aromaticity changes across the map are **allowed** — and
recorded in `plan.json` rather than left implicit, because they must be reviewable.

## What the environment must be

The environment is ONE matched build holding endpoint A. Endpoint B never gets its own box: two
independently solvated boxes are not atom-matched, and treating them as if they were builds a plan
that runs and is wrong.

The build record is **required**, and it is required to describe the two files given: the SHA-256 of
`built.xml` and `built.pdb` recorded in `built.log` are checked against the files themselves, and a
record that is not a *completed* `build-top` record is refused. It must also carry
`forcefield_record.ligand.nonbonded_compatibility`, which states the 1-4 scales the System
**applies** to the ligand — OpenMM applies its first `NonbondedForce` definition to every 1-4 pair,
and the OPC water XMLs write 5/6 as 0.833333, so the applied scale is read, never inferred. An
environment built before 0.6.0 does not record it and is refused with that message. The record's
`solvent.treatment` supplies the solvation — `explicit`, `implicit` or `vacuum` — which is likewise
never inferred from periodicity, because a vacuum leg and an implicit-solvent System are both
non-periodic.

Beyond that, the environment is refused when:

* the ligand selector names anything other than exactly one residue, or that residue's atom names
  and elements are not package A's **in package order** — it is then not endpoint A;
* the environment ligand does not carry package A's registered parameters, within the package's own
  comparison tolerance;
* the ligand masses are not the package's — hydrogen mass repartitioning, most likely. The appended
  endpoint-B atoms would need the same repartitioning, and this construction does not apply it;
* the System carries a force class outside `NonbondedForce`, `HarmonicBondForce`,
  `HarmonicAngleForce`, `PeriodicTorsionForce` on the ligand, or barostats, `CMMotionRemover` and
  `CMAPTorsionForce` elsewhere — implicit-solvent forces and custom forces are refused rather than
  carried unaltered through an alchemical change — or has more than one of the four ligand forces;
* a CMAP term touches the ligand, or a ligand atom is a virtual site;
* the `NonbondedForce` already has global parameters or parameter offsets: a scaled or alchemical
  System is not an environment;
* a constraint joins the ligand to another particle, or constrains a ligand pair that is not a bond
  (HAngles): only bond constraints can be carried to endpoint B;
* the ligand's constrained bonds are not exactly HBonds, AllBonds or None, so the policy cannot be
  applied to endpoint B. There is one further case: when *every* bond of the environment ligand is
  an X–H bond, HBonds and AllBonds constrain the same pairs and the build does not say which it
  was. That is only refused when endpoint B has a heavy–heavy bond whose treatment would depend on
  the answer — and the refusal says to build the environment from the other endpoint instead;
* the solvation and the System's periodicity disagree.

## Where the behaviour lives

The command surface adds no construction of its own. It resolves both packages through the one
package resolver, reads the environment, takes or proposes the map, and hands all of it to
`build_topology_plan`:

```text
resolve both packages      md_tools.ligands.catalog.resolve_package
read the environment       md_tools.alchemy.topology.Environment.from_files
the atom map               AtomMap.from_pairs / AtomMap.from_record, or propose_map
build and check the plan   md_tools.alchemy.topology.build_topology_plan
write it                   TopologyPlan.write
```

`tests/test_combine_topology_cli.py` asserts that the command writes exactly the plan the callable
layer builds, that `--check` creates nothing, that a proposed map reproduces its own plan, that an
edited proposal is refused, and that every accepted key does its job or is refused by name.

## Related

* [The methods index](../README.md) — the four protocols `build-md` generates today
* [The dataset contract](../../structure/project-data.md) — the schema-level authority for records
