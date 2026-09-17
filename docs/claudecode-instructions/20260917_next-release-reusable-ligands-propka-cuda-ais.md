# Next release implementation instruction — v6.0.0

Date: 2026-09-17
Target branch: dev
Status: implementation plan; no feature or validation result is claimed complete.

The user calls this release v6.0.0. The inspected repository is on the v0.5.x line.
Use this document's release label for planning; do not silently bump package versions or
create a release tag. Resolve the intended numbering (v6.0.0 versus v0.6.0) when preparing
the release, without blocking implementation.

## 1. Objective and boundaries

Make MD-tools prepare reproducible multi-component systems, reuse ligand parameters across
projects, assign protein protonation with PROPKA3, and run REST2 replicas and AIS paths
concurrently on shared CUDA GPUs. Demonstrate the preparation workflow through a 1TYL
protein–paracetamol tutorial with 10 ns conventional MD.

Read CLAUDE.md and the current implementation before changing code. This document extends
the existing design rather than introducing a second configuration or runtime framework.
Keep the four public work commands, Amber-like inputs, existing authoritative resolvers,
registration contracts, and standalone reference-export mechanism. Do not restore archived
rREST2. Do not merge to main or publish a release as part of this instruction.

Implement, validate, fix failures, and rerun the affected checks until their criteria pass.
Report unavailable hardware or unsupported chemistry as blocked evidence, never as a pass.
Do not repeatedly run unrelated expensive tests after the relevant risks are resolved.

## 2. Reusable ligand parameter packages

### Identity and storage

Separate three identities:

1. Compound catalog identity: use an external identifier such as CHEMBL112 for paracetamol,
   with paracetamol, acetaminophen, and CCD TYL stored as searchable aliases. Provide a
   documented local identifier for compounds without an external database entry.
2. Parameter identity: an immutable identifier bound to the exact chemical state and saved
   parameter artifact. A compound ID alone is not sufficient to authorize reuse.
3. Molecular instance: the particular ligand in a structure, selected by chain, residue
   number and insertion code, with model/assembly-copy identity where necessary.

Use the proposed shared catalog location:

    $MD_DATA/parameters/ligands/<compound-id>/<parameter-id>/
        molecule.sdf
        parameters.ffxml
        metadata.json

This is a parameter catalog, separate from the existing registered simulation-dataset paths.
Integrate creation/reuse into build-top and registration into data-register without adding a
fifth public command. Choose minimal strict configuration fields and document them.

The SDF must define the exact chemical state, including explicit hydrogens, formal charges,
bond orders, stereochemistry and an atom identity convention. Keep separate records for
different tautomers/protomers even if an external identifier or normalized InChI merges them.
Coordinates from this reference SDF must not replace the ligand's crystallographic pose.

Metadata must include aliases, chemical-state identity, net formal charge, actual partial
charges and their atom correspondence, force-field/version, charge method/backend/model,
relevant software versions, source provenance and artifact hashes. Define the hashing and
canonicalization scheme; never infer parameter identity from a directory label alone.

Store the actual reusable parameters, not just a recipe that reruns charge assignment.
Names of generated atom types and templates must avoid collisions between packages.
Verify that the chosen ffxml representation reproduces all supported force terms and
nonbonded conventions. Refuse unsupported conversions rather than losing terms.

### Reuse behavior

Parameterize once. Ligand-only and protein–ligand builds must load that same artifact without
rerunning charge generation. Copy the small package into each build/export so it remains
self-contained after relocation or loss of the shared catalog.

For the existing paracetamol dataset, inspect its artifacts first. If recovering parameters
from the unscaled saved System is needed, use the retained topology/SDF to establish identity
and validate the extraction. Never extract a REST2-scaled Hamiltonian as the physical ligand
parameters. Do not mutate an already registered simulation dataset.

Reusing parameters does not require identical ligand coordinates. It does require identical
chemical state and compatible force-field/nonbonded conventions in the assembled system.
Reject incompatible protein/ligand parameter conventions with an actionable explanation.

## 3. Multiple chains, distinct ligands and repeated copies

Support multiple protein chains, multiple ligand species, and repeated instances of a species.
Preserve intended chain termini, disulfides, residue IDs, ligand poses and relevant ions.
Do not treat every chain boundary or nearby atom pair as a new covalent bond.

Provide an explicit ligand-instance-to-package mapping. Illustrative proposed configuration
only; these are not asserted 1TYL residue numbers:

    ligands:
      - select: {chain: A, resid: "201", insertion_code: ""}
        parameters: CHEMBL112/param_abc123
      - select: {chain: B, resid: "201", insertion_code: ""}
        parameters: CHEMBL112/param_abc123
      - select: {chain: A, resid: "202", insertion_code: ""}
        parameters: MOL_xyz789/param_def456

Preserve PDB-compatible residue names such as TYL; do not put CHEMBL112 into a three-character
PDB residue-name field. OpenMM template matching uses elements/connectivity rather than the
catalog name. Explicitly select templates per OpenMM Residue through residueTemplates when
needed, including during preparation calls that perform force-field matching.

Resolve selectors uniquely before building. Validate each ligand against its reference
chemical graph and save an explicit atom mapping. Do not assume PDB/SDF atom order matches.
OpenMM connectivity matching alone cannot establish all bond orders, stereochemistry or
chemical-state distinctions. Preserve parameter-to-atom correspondence under reordering and
handle symmetry consistently; reject unresolved chemically consequential ambiguity.

Record both original selectors and resolved topology indices. Avoid confusing mmCIF label
chain IDs with author chain IDs. Assembly expansion must produce distinguishable instances.
Covalently attached ligands are not implicitly supported by a single-residue ligand package;
refuse them clearly unless an explicit validated representation is implemented.

## 4. PROPKA3 and protonation preparation

### Prediction and assignment

Add a PROPKA3 preparation option, exercised by the new tutorial, with target pH defaulting
to 7.0. Preserve the existing OpenMM-only mode as an explicitly recorded choice.
When PROPKA is requested, an unavailable dependency or failed prediction is an error, not
a silent fallback to default residue states.

Run on the prepared protein–ligand complex with relevant ions retained. Validate which
ligand/ion groups the installed PROPKA actually recognizes; retain warnings and report
unsupported groups. Do not claim comprehensive metal-site modeling merely because ions
were present in the input.

PROPKA predicts pKa values. MD-tools must separately translate supported predictions into
force-field-compatible residue variants. Document the deterministic assignment rule and
flag near-pH or coupled/ambiguous cases. Unsupported predicted variants must be reported,
not silently converted into another state. Handle termini and disulfides consistently.

For ligands, retain supported predictions in the record. For an existing parameter package,
never change protonation/tautomer or hydrogens while keeping its old parameters. If the
prediction suggests another state, warn and explain that a separate parameter package is
needed. For a newly parameterized ligand, establish the explicit chosen chemical graph
before generating charges; PROPKA alone is not a general ligand tautomer enumerator.

### Histidine proximity warnings and explicit overrides

PROPKA must not be presented as resolving neutral histidine HID versus HIE. Keep the existing
OpenMM hydrogen-bond heuristic as the documented default for neutral histidines without an
override, while honoring a protein ionization assignment that requires HIP.

Detect histidines near any ligand or ion using a configurable minimum heavy-atom distance,
proposed default 5 Angstrom. Print a CLI warning with chain/residue/insertion code, neighbor
identity, measured distance, predicted pKa when available, final variant and its source.
For metal neighbors, report ND1–metal and NE2–metal distances separately.

Proximity is a screening criterion, not proof of coordination. Do not automatically impose a
tautomer based only on the 5 Angstrom threshold. Permit explicit per-residue HID/HIE/HIP
overrides and report if an override supersedes a prediction. Warnings alone do not block
an otherwise valid build. Unknown selectors and incompatible variants do block it.

Preserve ligand chemical states during hydrogen addition: the current protein/complex
hydrogen-deletion path must not indiscriminately remove registered ligand hydrogens.
Add protein hydrogens using explicit resolved assignments, and ensure later solvation or
reloading does not overwrite them.

Save the predictor version/settings, input digest, raw prediction output, unsupported-group
warnings, final per-residue states, assignment sources, overrides and proximity warnings.
Bind assignments to residue identities, not only to a positional list. Save the resulting
prepared structure. Ordinary cMD holds these states fixed; this is not constant-pH MD.

## 5. Concurrent CUDA execution for REST2 and AIS

Audit existing MPI/device distribution before adding code. Reuse the existing MPI authority,
platform resolver, output ownership and fail-collectively behavior.

REST2 must allow one worker per replica with multiple workers assigned to each GPU.
Required demonstrations: 4 replicas on 1 GPU and 12 replicas on 4 GPUs. All replicas propagate
between exchange boundaries concurrently as GPU resources permit, then coordinate exchanges.
A single-process loop integrating each replica in sequence is not evidence of concurrency.

AIS must distribute independent global path IDs over multiple processes sharing the GPUs.
Each path remains sequential in its own switching coordinate. Preserve path-ID-based seeds,
file names and resume identity regardless of worker count. Do not reintroduce Context reuse
that changes path behavior or seed independence without separate evidence.

Use NVIDIA MPS for the shared-GPU execution option. Document environment setup, daemon
lifecycle and launch commands; distinguish requested MPS, detected/verified MPS and unknown
status. Process assignment alone does not prove that MPS is active. Do not silently terminate
an existing shared MPS service. Keep MPS/device settings in machine/execution configuration,
not in the scientific Hamiltonian.

Respect CUDA_VISIBLE_DEVICES and use local-rank-aware assignment on each node. Record rank,
local rank, physical device identity when available, precision, versions and relevant MPS
settings. Do not hard-code workstation GPU numbers.

Validate concurrent execution through overlapping worker activity plus device/MPS evidence
or profiling, not GPU utilization alone. Compare sequential, concurrent without MPS, and
concurrent with MPS under matched scientific settings and workload. Report total throughput,
wall time and GPU memory. Do not promise a speedup for every system; a 22-atom system is
particularly sensitive to launch, reporting and synchronization overhead.

Preserve exchange acceptance calculations, trajectory/state conventions, checkpoint
transactions and collective failure handling. Concurrency must not change the protocol.

## 6. AIS Langevin protocol and OpenMMTools consistency

The goal is a precisely defined, reproducible OpenMMTools-compatible switching option.
The exact new splitting and default were NOT finalized in the discussion. Do not claim
that O { V R H R V } O has been selected, or silently replace LangevinMiddle.

Before implementation, pin an OpenMMTools version/commit and identify the exact reference
integrator. Write a short protocol decision record specifying splitting, substep durations,
lambda-update placement, thermostat, constraints, work definitions and endpoint convention.
Keep the existing middle-propagation scheme available and preserve its meaning.

Distinguish an ordinary nonequilibrium Langevin splitting from a splitting containing
Metropolis braces. If evaluating O { V R H R V } O, inspect exactly what is accepted/rejected
and restored, including coordinates, velocities, lambda and work variables. Do not describe
the braces as an automatically valid correction to an AIS estimator without deriving and
testing the associated weighting.

Maintain the existing two-System interpolation V(lambda)=(1-lambda)V0+lambda V1, fixed volume,
compatible particles/masses/constraints, and exact frozen-coordinate protocol work.
Retain PME and long-range correction contributions. Observation zero has zero work.

Name protocol work and any shadow work separately, state their units/signs, and state which
quantity enters each estimator. Do not calculate shadow work naively from LangevinMiddle
half-step velocities. Keep saved coordinates, lambda, cumulative work and endpoint-energy
observations aligned, including the existing Hummer–Szabo outputs.

Compare matched reference implementations on an analytic system and small molecular cases.
Different RNG implementations need statistical rather than bitwise trajectory comparison.
Predeclare tolerances and sampling budgets; verify work arithmetic at fixed coordinates,
free-energy estimates against a known answer, timestep dependence, ESS and cost.
Pin the new scheme into run fingerprints/checkpoints and reject incompatible continuations.
Preserve the current documented CUDA mid-path restart limitations.

## 7. 1TYL protein–paracetamol tutorial

Scope: preparation, equilibration and 10 ns production cMD only. No REST2 or AIS production
campaign is required for this tutorial.

Use deposited paracetamol (residue name TYL; compound CHEMBL112). Choose and document the
biological assembly explicitly: the asymmetric unit and complete insulin hexamer are not
interchangeable. Record structure/assembly source and hashes, alternate-conformer handling,
missing atoms/residues, retained waters/ions and disulfides.

If zinc is retained, document a supported parameter model and inspect coordinating
histidines. Ordinary monatomic ion parameters do not by themselves establish accurate
metal coordination. Do not silently delete zinc or choose an assembly simply to make a
build succeed. Record preparation choices and their scientific limitations.

Reuse the registered ligand-only paracetamol package without regeneration and preserve its
bound heavy-atom pose. Show protein/ligand mapping, PROPKA results, histidine warnings and
any overrides. Use explicit documented force-field/solvent settings and integer-step stages.

Publish runnable configurations/commands and explain registration plus standalone export.
Report actual production length, temperature/density behavior, protein and ligand structural
diagnostics and zinc-site distances if applicable. Ten nanoseconds demonstrates workflow
operation and short-time stability, not binding affinity or equilibrium convergence.

## 8. Acceptance criteria and evidence

| Area | Required evidence |
| --- | --- |
| Parameter reuse | Same package digests and ligand per-atom/bonded/nonbonded parameters in ligand-only and complex builds; no charge-generation call on reuse. Validate isolated-ligand energies/forces at common coordinates, not whole-system energies against each other. |
| Mapping | Repeated copies and a distinct second ligand map correctly; reordered atoms give equivalent physical parameters/energies; ambiguous selectors, mismatched chemical states and incompatible packages are rejected. |
| Protonation | Real PROPKA integration fixture, deterministic prediction-to-variant mapping, explicit overrides, unsupported-group reporting, near-ligand/ion warnings and preserved ligand hydrogens. An ordinary nonproximal histidine does not spuriously trigger proximity warnings. |
| CUDA REST2 | Actual 4-on-1 and 12-on-4 runs; concurrency/MPS evidence; valid state-to-walker permutations, energy-based exchange checks, output integrity, restart/extension and collective failure behavior. |
| CUDA AIS | Multiple simultaneous paths on shared GPUs; stable path identity/seeds across scheduling; complete work/frame alignment; restart without duplicated committed rows and without changing scientific identity. |
| AIS reference | Pinned reference and documented splitting; independently checked work; analytic free-energy result within predeclared uncertainty; timestep and ESS/cost comparison; molecular implicit and explicit/PME checks. |
| Tutorial | Completed 10 ns cMD after equilibration; all inputs and preparation decisions recorded; reused ligand package; analysis and limitations reported. |
| Export | Relocated standalone bundle reproduces the built Hamiltonian and executes the supported protocol with OpenMM and explicitly declared output dependencies, without MD-tools or a parameterization/prediction package at runtime. |
| Regression | Relevant existing tests, CUDA lanes and installed-wheel checks pass; strict configuration and continuation contracts remain intact. |

For the standalone export, a serialized System reproduces the already built simulation
without rerunning PROPKA/OpenFF. Distinguish this from rebuilding from raw PDB, which
requires those preparation dependencies. Carry saved REST2 states and all mappings/records.

For each criterion report the commit, environment, exact command, inputs, expected tolerance,
observed result and evidence location. Do not claim old hardware reservations remain valid.
Acquire currently available resources. Record hardware-blocked tests explicitly.

Recommended implementation order: (1) package identity/reuse and multi-component mapping,
(2) PROPKA and warnings, (3) tutorial build and cMD, (4) concurrent CUDA/MPS validation,
(5) AIS reference decision and implementation, (6) registration/export and release evidence.
Start the 10 ns run once preparation checks pass while unrelated implementation continues.

## 9. Future work — explicitly outside this release

### Selected-residue REST2

Support user-facing selections such as scaled-SC-resid = [] for side chains and
scaled-BB-resid = [] for backbone phi/psi. Define chain-aware residue identity, exact atom
and torsion membership, overlap rules, cross-boundary interactions and how this interacts
with existing unscaled amide/aromatic/double-bond/improper rules. Distinguish torsion-only
selection from atom-based nonbonded scaling. Do not promise that all of this is already
represented by a residue list. Preserve selections in saved-state provenance.

### Thermodynamic integration (TI)

Add explicit lambda schedules, equilibrium window sampling, dU/dlambda observables,
quadrature, uncertainty and convergence diagnostics. Define supported Hamiltonians and
consistent long-range contributions. Reuse existing parameter/mapping identities.

### Free-energy perturbation (FEP)

Define supported transformations and estimators before exposing commands: endpoint and
intermediate-state energy evaluation, BAR/MBAR where appropriate, overlap/effective-sample
diagnostics and uncertainty. Atom mapping, softcore, restraints, charge-changing corrections
and topology-changing transformations require explicit designs and validation. Current
linear two-System AIS must not be advertised as general alchemical FEP support.

### More advanced protonation/tautomer preparation

Future options may optimize coupled hydrogen-bond networks and metal-aware tautomers or
import assignments from external preparation software. Automatic general ligand tautomer
enumeration, quantitative metal-site pKa treatment and constant-pH MD are not in this release.
The present predictor/override records should allow these additions without changing the
meaning of existing parameter packages.

## 10. Reference entry points

- Repository guidance: CLAUDE.md; docs/README.md.
- Existing build/preparation: src/md_tools/openmm/system.py, builders.py and builder_defaults.py.
- Existing AIS interpolation: src/md_tools/ais/two_state.py; inspect current integrator/runtime modules.
- PROPKA: https://propka.readthedocs.io/en/latest/command.html
- PROPKA hydrogen handling: https://propka.readthedocs.io/en/latest/_modules/propka/protonate.html
- OpenMM residue templates: https://docs.openmm.org/latest/api-python/generated/openmm.app.forcefield.ForceField.html
- OpenMM hydrogen addition: https://docs.openmm.org/latest/api-python/generated/openmm.app.modeller.Modeller.html
- OpenMMTools reference source: https://github.com/choderalab/openmmtools/blob/main/openmmtools/integrators.py
- NVIDIA MPS: https://docs.nvidia.com/deploy/mps/
- Structure: https://www.rcsb.org/structure/1TYL
- Compound: https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL112

Pin versions actually used during implementation; these links alone are not a reproducibility record.
