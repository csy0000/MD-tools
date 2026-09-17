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
projects, assign protein protonation with PROPKA3, run REST2 replicas and AIS paths
concurrently on shared CUDA GPUs, and give AIS a second switching schedule and a correct
evenly spaced source-frame selection. Demonstrate the preparation workflow through a 4A9K
protein–paracetamol tutorial and a 1BRS barnase–barstar protein–protein tutorial, each with
10 ns conventional MD.

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
only; these are not asserted 4A9K residue numbers:

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

### CPU cores and GPU placement

The current placement deals ranks over visible devices in turn (`rank % n_devices` in
`md_tools.remd.engine.select_device_for_rank`). Without MPS, two ranks on one GPU are
time-sliced, and a synchronous ladder then runs at the pace of that shared GPU. Replace it:

- CPU cores. The number of CPUs available to the launch must be an integer multiple of the
  number of concurrent workers (REST2: the replica count; AIS: the worker count). For 4
  replicas, 8 or 12 cores are accepted and 5 is refused in the preflight, before any output,
  naming the arithmetic that would fix it. Count the CPUs the launch may actually use
  (affinity/cgroup), not a number a configuration claims, and bind each worker to its own
  equal block of cores. Record the count and the binding.
- GPU count. Fewer GPUs than replicas is allowed, including an uneven split such as 4
  replicas on 3 GPUs. MPS is required whenever a GPU hosts more than one worker; its absence
  is refused, never worked around.
- Balanced placement. Place workers by measured per-device throughput rather than in turn,
  so the slowest device carries the least load. Record the measurement and the resulting
  worker-to-device map. Placement must never change the scientific protocol, and a ladder's
  state index stays the state's, not the device's.

## 6. AIS switching schedules and source-frame selection

Keep the current integrator (LangevinMiddle) and its meaning. An OpenMMTools-compatible
splitting is NOT part of this release (see section 9).

### Schedules

`V(λ) = (1−λ)V0 + λV1` is unchanged; only how λ follows the normalised switching progress
`t = step / switching_steps` changes. Two kinds:

- `linear`: `λ = t` (the current and default behaviour).
- `tau-linear`: `λ = [(1−τ₀+τ₀t)² − (1−τ₀)²] / [1 − (1−τ₀)²]`, with τ₀ the tau of V0.
  With V1 unscaled, this makes the solute–solute prefactor of the mixture, `(1−λ)(1−τ₀)² + λ`,
  equal to REST2's `(1−τ)²` along a tau that is linear in t. The solute–environment prefactor
  then does NOT follow `(1−τ)` exactly; no single λ can match both, and the documentation
  must say so.

AIS itself must not read REST2 records. `build-md`, which knows τ₀ from
`build/<method>/scaler.yaml`, resolves `tau-linear` into an explicit λ value per parameter
update and writes it into the resolved configuration; the runtime only consumes λ values.
Refuse `tau-linear` when V0 is not a saved scaled state or V1 is not its unscaled source.
λ must be monotone from exactly 0 to exactly 1. Work stays the frozen-coordinate finite
difference `ΔW_j = V(λ_{j+1}, x_j) − V(λ_j, x_j)`, which for the linear mixture is
`(λ_{j+1} − λ_j)(V1 − V0)(x_j)` for any schedule. Record the schedule kind, τ₀ and a digest of
the λ table in `AIS_run.json` and the checkpoint fingerprint; a continuation under a different
schedule is refused by name.

### Evenly spaced source frames

`selection: evenly_spaced` currently floors the stride (`len(eligible) // count`), so 64 paths
from 95 eligible frames use frames 5–68 and never the rest of the window. Spread the picks over
the whole window with deterministic fractional spacing, keep distinct frames, and keep the
existing refusal when there are more paths than frames. The selected frames are part of the run
identity, so an existing run directory with the old selection is refused, not silently mixed.

## 7. Tutorials: 4A9K protein–paracetamol and 1BRS barnase–barstar

### 4A9K protein–paracetamol

Changed by the user on 2026-09-17: the protein–paracetamol tutorial uses **4A9K** (CREBBP
bromodomain with N-(4-hydroxyphenyl)acetamide, residue TYL; X-ray, 1.81 Å) instead of 1TYL, so
that no metal site has to be parameterized. The 1TYL insulin-hexamer and MCPB.py zinc-site work
is out of this release (section 9).

Scope: preparation, equilibration and 10 ns production cMD only. No REST2 or AIS production
campaign is required for this tutorial.

#### Structure facts (from the mmCIF) and required decisions

- Assembly 1: chain A (115 residues), 1 TYL, 2 EDO, 1 SCN, 150 waters.
- Assembly 2: chain B (112 residues), 1 TYL, 1 K+, 117 waters.
- Neither has a chain break. Each has 3 residues with missing side-chain atoms plus a C-terminal
  OXT (built under `input.missing_atoms: add`) and 4 residues with alternate conformations.

Choose ONE biological assembly explicitly and record why, with the source file, assembly
definition and hashes. Record alternate-conformer handling, rebuilt atoms, and every retained
or removed water and ion. EDO (ethylene glycol) and SCN (thiocyanate) are crystallisation
additives: either remove each explicitly, recorded as a preparation decision, or keep it with its
own parameter package. Never drop them silently, and never pick an assembly only because it
builds more easily. A retained K+ uses TIP3P-compatible Amber ion parameters and needs no
site-specific model.

| Component | Tutorial choice |
| --- | --- |
| Protein | Amber ff14SB |
| Water | TIP3P |
| Paracetamol | Existing registered Sage/charge parameter package, unchanged |
| Ions (retained or bulk) | TIP3P-compatible Amber ion parameters |

Reuse the registered ligand-only paracetamol package (CHEMBL112) without regeneration and
preserve its bound heavy-atom pose. Show protein/ligand mapping, PROPKA results, histidine
warnings and any overrides. Use explicit documented force-field/solvent settings and integer-step
stages.

Publish runnable configurations/commands and explain registration plus standalone export.
Report actual production length, temperature/density behavior, and protein and ligand structural
diagnostics (for example protein CA RMSD, ligand heavy-atom RMSD in the binding site, and the
ligand's key contacts). Ten nanoseconds demonstrates workflow operation and short-time stability,
not binding affinity or equilibrium convergence.

### 1BRS barnase–barstar

Scope: the same as 4A9K — preparation, equilibration and 10 ns production cMD — for a
protein–protein complex with no ligand package. It exercises multiple chains, PROPKA
assignment and histidine warnings at an interface without a small molecule.

Inspect the deposited entry before building: record how many complexes the asymmetric unit
holds and select one barnase–barstar pair explicitly by chain, and record any engineered
mutations, missing residues, retained waters and ions. Preserve chain termini; do not create
bonds across the interface. Report the interface structure over the production run (for
example interface contacts or RMSD of the complex), temperature/density behaviour and the
limitations of a 10 ns run.

## 8. Acceptance criteria and evidence

| Area | Required evidence |
| --- | --- |
| Parameter reuse | Same package digests and ligand per-atom/bonded/nonbonded parameters in ligand-only and complex builds; no charge-generation call on reuse. Validate isolated-ligand energies/forces at common coordinates, not whole-system energies against each other. |
| Mapping | Repeated copies and a distinct second ligand map correctly; reordered atoms give equivalent physical parameters/energies; ambiguous selectors, mismatched chemical states and incompatible packages are rejected. |
| Protonation | Real PROPKA integration fixture, deterministic prediction-to-variant mapping, explicit overrides, unsupported-group reporting, near-ligand/ion warnings and preserved ligand hydrogens. An ordinary nonproximal histidine does not spuriously trigger proximity warnings. |
| CUDA REST2 | Actual 4-on-1 and 12-on-4 runs; concurrency/MPS evidence; valid state-to-walker permutations, energy-based exchange checks, output integrity, restart/extension and collective failure behavior. |
| CUDA AIS | Multiple simultaneous paths on shared GPUs; stable path identity/seeds across scheduling; complete work/frame alignment; restart without duplicated committed rows and without changing scientific identity. |
| CPU/GPU placement | Core counts that are not a multiple of the worker count refused before output; uneven GPU splits (4 replicas on 3 GPUs) accepted only with MPS verified; placement from measured throughput recorded; throughput compared with the old round-robin under matched settings. |
| AIS schedules | `linear` unchanged bit-for-bit in its λ values; `tau-linear` λ table equal to the formula, monotone, exactly 0 and 1 at the ends; the mixture's solute–solute prefactor equal to `(1−τ)²` at every update; frozen-coordinate work checked against independent end-state energies; schedule change on continuation refused. |
| Frame selection | `evenly_spaced` covers the whole eligible window with distinct frames; more paths than frames still refused; old-selection run directories refused on continuation. |
| Tutorials | 4A9K and 1BRS each complete 10 ns cMD after equilibration; all inputs and preparation decisions recorded, including the assembly choice and the handling of EDO, SCN, K+ and alternate conformers; 4A9K reuses the ligand package; analysis and limitations reported. |
| Export | Relocated standalone bundle reproduces the built Hamiltonian and executes the supported protocol with OpenMM and explicitly declared output dependencies, without MD-tools or a parameterization/prediction package at runtime. |
| Regression | Relevant existing tests, CUDA lanes and installed-wheel checks pass; strict configuration and continuation contracts remain intact. |

For the standalone export, a serialized System reproduces the already built simulation
without rerunning PROPKA/OpenFF. Distinguish this from rebuilding from raw PDB, which
requires those preparation dependencies. Carry saved REST2 states and all mappings/records.

For each criterion report the commit, environment, exact command, inputs, expected tolerance,
observed result and evidence location. Do not claim old hardware reservations remain valid.
Acquire currently available resources. Record hardware-blocked tests explicitly.

Recommended implementation order: (1) package identity/reuse and multi-component mapping,
(2) PROPKA and warnings, (3) tutorial builds and cMD, (4) concurrent CUDA/MPS validation with
the core and placement rules, (5) AIS schedules and frame selection (independent of 1–4, can
start at once), (6) registration/export and release evidence.
Start the 10 ns run once preparation checks pass while unrelated implementation continues.

## 9. Future work — explicitly outside this release

### OpenMMTools-compatible AIS Langevin splitting

Deferred by the user (2026-09-17). When picked up: pin an OpenMMTools version and the exact
reference integrator; write a protocol decision record (splitting, substep durations, λ-update
placement, thermostat, constraints, work definitions, endpoint convention); keep LangevinMiddle
available with its meaning; name protocol work and shadow work separately; do not treat
Metropolis braces as a valid AIS correction without deriving and testing the weighting; compare
against matched references statistically on analytic and molecular cases.

### Bidirectional switching and alchemical transformations

Reverse switching with BAR/Crooks estimators, atom mapping and softcore are not in this
release.

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

### General metal-site preparation

General automated MCPB.py/QM orchestration, transferable metal-site libraries, alternative
12-6-4 or dummy-atom models, and coordination-exchange simulations are deferred. So is the
1TYL insulin-hexamer tutorial (biological assembly 3, site-specific MCPB.py zinc parameters
fitted with QM), which the 4A9K tutorial replaced in this release; no QM program is installed on
the development machine.

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
- Structure: https://www.rcsb.org/structure/4A9K
- Structure: https://www.rcsb.org/structure/1BRS
- Compound: https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL112

Pin versions actually used during implementation; these links alone are not a reproducibility record.
