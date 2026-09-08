# Implement peptide-like macrocycles with verified mbondi3 corrections

## Goal
Implement this change on MD-tools dev, validate it, and push the code, tests and report.
Read CLAUDE.md and preserve unrelated work. This supersedes the completed RGDfV
integration task; retain docs/integration/rgdfv-rest2.md as the historical record.

Support head-to-tail cyclic peptides composed of canonical amino-acid side chains,
including D stereoisomers, parameterized as a whole molecule with Sage/AM1-BCC.
RGDfV is the concrete integration case. Do not expand into a general modified-peptide
framework, solvent-model comparison, new force field or longer convergence study.

## 1. One classification in the existing configuration
Add solute.kind with exactly peptide, peptide-like, ligand. Keep existing configuration
locations, e.g. this YAML .config:
    solute:
      kind: peptide-like
      ligand_forcefield: sage-2.2.1
      ligand_charge_method: am1bcc
    solvent:
      model: GBn2
    constraints:
      type: HBonds
    hydrogen_mass_repartitioning:
      enabled: false

peptide uses the existing protein route; ligand uses the existing whole-molecule route;
peptide-like uses that SAME whole-molecule parameterization route plus validated peptide
chemistry mapping. It must not load ff14SB or replace Sage charges/bonded terms.
Do not introduce a separate parameterization implementation.

Backward compatibility:
- Explicit legacy peptide:true maps to kind:peptide; false maps to kind:ligand.
- Neither field specified retains the current peptide default.
- Resolve aliases before defaults so an injected peptide:true cannot override kind.
- Both explicitly supplied: accept only the exact compatible alias pair
  (peptide/true or ligand/false); otherwise refuse with a migration message.
  peptide-like has no boolean alias; omit the legacy key.
- Persist one authoritative classification; any derived compatibility field is never
  independently authoritative. Existing saved configs must remain readable.
- Unknown values and incompatible input formats fail before publication.
Inspect build/top.py, openmm/system_config.py and route consumers, examples and logging.
Keep actual force-field resource selection separate from chemical classification.

## 2. Validated molecular map
Reuse RDKit and the retained SDF graph; add one reusable mapping implementation.
Do not infer chemistry from a single UNL/custom residue name or hard-code RGDfV indices.
For v1 require an unambiguous head-to-tail alpha-peptide backbone cycle and canonical
side-chain identities. Include glycine, proline and D stereoisomers without equating
L/D universally to S/R (cysteine is an exception). Require specified stereochemistry
where applicable. Preserve protonation, formal charges, all bonds and atom order.

Map source graph -> prepared SDF -> System solute indices explicitly, accounting for
hydrogen addition and explicit-solvent particles where applicable. Verify connectivity,
bond orders and stereochemistry through conversions; equal atom counts alone are insufficient.
Record residue identities, stereochemistry, protonation, backbone N/CA/C/O indices,
cyclic links, and the affected side-chain atom sets.
Equivalent symmetric matches may be accepted only if all affected atom sets agree.
Reject ambiguous decomposition or unsupported modified side chains with a useful reason;
the generic ligand route remains available explicitly. Do not guess a partial mapping
while claiming complete peptide-like support.

Use the same map for meaningful cyclic phi/psi/omega definitions and radius assignment.
Reuse existing omega classification/scaler; cross-check its output against the map.
Proline-like exclusions/scaling retain the existing scientific policy.
Retain existing general ligand behavior. No automatic new restraints or forces for CVs.

## 3. Correct radii before System construction
The current implicit builder calls ParmEd changeRadii(..., "mbondi3"), whose Arg/Asp/Glu
rules depend on residue/atom names. A single ligand residue misses these rules.
For mapped peptide-like chemistry, assign the equivalent established corrections after
baseline radius assignment but BEFORE createSystem/serialization:
- deprotonated Asp/Glu side-chain carboxylate oxygens: 1.40 angstrom each;
- positively charged Arg guanidinium hydrogens corresponding to canonical HH*/HE*:
  1.17 angstrom each.
RGDfV should have TWO affected Asp oxygens and FIVE affected Arg hydrogens.
Do not change carbonyl oxygens, backbone NH hydrogens or other atoms.
Do not apply charged-group corrections to neutral protonation variants.
The supported head-to-tail cycle has no terminal OXT correction.

Verify these rules against the installed ParmEd implementation and record its version.
Reference: https://parmed.github.io/ParmEd/html/_modules/parmed/tools/changeradii.html
Use chemical identities/protonation, not arbitrary residue renaming or guessed radii.
If a selected radius/model policy is inapplicable, do not quietly apply mbondi3 anyway.
Explicit-solvent builds must not acquire GB terms or radius corrections.

Record requested radius policy, actual assignment method, correction atom indices and
old/new intrinsic radii with units, molecular-map digest and corrected-atom count.
Replace name-only coverage claims for this route with measured evidence.
Report intrinsic radii separately from OpenMM offset radii and effective Born radii.
The serialized CustomGBForce stores transformed parameters: verify using the installed
GBn2 offset (normally 0.0195141 nm), not by comparing its field directly to 0.14 nm.
Check corresponding dependent parameters are rebuilt consistently.

All solute charges, valence and LJ/exception parameters, masses and constraints must
match an otherwise identical ligand build. Only the intended GB parameters change.
Keep the current GBn2 scaling by (1-tau), and label Sage+GBn2 as a hybrid whose
conformational accuracy is not established by implementing these corrections.

## 4. Tests with concrete pass criteria
First demonstrate the missed corrections on the current RGDfV ligand build.
Add focused regressions rather than assertions that merely mirror implementation.

A. Configuration:
test each new kind, legacy booleans, absent fields, matching aliases, conflicts,
unknown values and config round trips; peptide-like demonstrably uses Sage.
B. Chemistry:
RGDfV gives five residues, D-Phe, the five cyclic links and correct charged groups.
Use atom-permuted/relabelled inputs to prove index/name independence.
Add compact fixtures exercising Glu, Pro/Gly and D/L handling (including Cys), neutral
side-chain controls, and rejection of ambiguous/unsupported chemistry.
C. Radii:
prove exactly 2 O and 5 H corrections for charged RGDfV, with all other intrinsic
radii unchanged from the legacy ligand baseline. Independently compare these atom
sets to a trusted canonical-residue ParmEd mbondi3 assignment; never derive both
expected and observed values through the new mapper.
Neutral-group controls do not receive the charged correction.
Serialize/reload the actual System; reconstruct intrinsic radii from offset fields
with explicit units and check abs tolerance 1e-8 nm. Check finite energies and forces.
D. Hamiltonian:
with identical charges and coordinates, non-GB parameters remain unchanged; compare
GB energies on several configurations, without imposing a sign or arbitrary minimum
magnitude. At tau=0 reproduce the corrected base System. At tau=0.25 and 0.5
retain existing omega/eligible-torsion/nonbonded/GB scaling invariants using existing
repository tolerances, specified before inspecting results.
E. Integration:
both md-run and generated entry points consume the corrected serialized System and
map without recomputing a different assignment. Map/SDF/provenance survive data inventory.
Legacy peptide and ligand representative builds retain their original Hamiltonians.
An explicit peptide-like smoke build retains Sage and has no GB force.

Run the fast suite, affected real CUDA/MPI lanes and wheel install outside the checkout.
Include cMD, REST2, rREST2 and AIS smoke coverage for the shared corrected System/scaler;
reuse existing test lanes instead of four new scientific projects.
Run full GPU/slow coverage if shared scientific runtime or restart code changes.
CUDA is mandatory evidence; CPU graph/parameter checks are legitimate but not a
substitute. Record skips and blockers honestly.

## 5. Corrected RGDfV sibling integration
Use ../RGDfV-REST2 relative to the actual checkout. Preserve the existing project and
registered dataset; create a distinctly named corrected-mbondi3 dataset/run directory.
Never resume old checkpoints with the changed Hamiltonian or alter old registered files.
Reuse verified molecular input and AM1-BCC charges/cache ONLY with matching identity,
stereochemistry, protonation and atom mapping, recording cache origin and charge digest.
Otherwise recalculate actual AmberTools AM1-BCC; do not silently substitute ELF10/NAGL.

Install the new committed wheel in the sibling environment, record commit/import path.
After chemistry, radii and force-level checks pass:
- six states tau=0,0.1,0.2,0.3,0.4,0.5 at 300 K;
- HBonds, 2 fs, HMR off, no SASA, nonperiodic GBn2;
- minimize then 100 ps per-state equilibration;
- 1 ns production/state, exchange every 10 ps;
- all 15 backbone CVs every 1 ps, trajectory/checkpoint every 10 ps;
- interrupt once after >=200 ps and resume to the original production budget.
Use available CUDA devices and six MPI ranks without disturbing unrelated jobs.
Verify final grids (1001 CV rows/state), completion, cyclic integrity, mapped CVs
against matching coordinates, omega force-level exclusion, and unchanged assignment
across resume. Reuse the established uninterrupted-reference comparison if affordable;
do not claim bitwise reference equality unless actually measured.
Report acceptance and Arg/Asp contact diagnostics descriptively, with no convergence claim
or acceptance threshold. Do not tune the ladder during this test.

Assemble required metadata and validation reports BEFORE registration. Use existing
identity/storage configuration, supported data-register dry run then transaction and
verify-only (include required project/data/year arguments). Use a new dataset identity
containing a corrected-mbondi3 distinction. Verify hashes and reopen CVs/trajectories
through the project symlink. Do not invent a storage destination or license.

## 6. Validate -> fix -> revalidate -> report
For every failing required check: retain the command and diagnostic, identify the cause,
add a focused regression, implement the fix and rerun that check and its dependent gates.
Do not weaken tolerances, rename failures as passes or remove failing tests.
Stop only on a concrete unavailable dependency/hardware or unresolved chemical ambiguity,
with completed changes and precise remaining blocker recorded.
Avoid unrelated backlog fixes, broad refactors, new commands and automatic long production.

Update configuration examples/help, scientific-defaults and a concise migration note.
Add docs/integration/rgdfv-peptide-like-mbondi3.md with actual corrected atom/radius
table, engine pin, commands, test counts, CUDA evidence, integration/registration result
and limitations. Link the historical report rather than rewriting its old-model result.
Push code/tests/docs to dev. Final handoff must distinguish implemented, measured,
not run and blocked. The goal is an actual verified fix, not another instruction-only commit.
