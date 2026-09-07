# RGDfV macrocycle: bounded REST2 integration and data registration

## Goal and scope

Execute one real integration project in a sibling folder of MD-tools:
../RGDfV-REST2/

Prove three things:
1. The five backbone omega central bonds are excluded from REST2 torsion scaling.
2. The whole macrocycle can be parameterized using Sage 2.2.1 and AM1-BCC and run in implicit solvent.
3. A completed dataset meets the current MD-tools data contract, can be deposited under the user's actual MD_DATA root, and can be read and verified there.

This is the next project integration experiment, not another exhaustive engine audit.
Read CLAUDE.md. Use the tested engine commit
443fc736f3a8f426414e244a50a1a438d2d8a497 initially.
Do not reset the user's MD-tools checkout to that commit: install it from an isolated checkout/wheel.
Record the actual installed commit, package import origin and environment. Preserve unrelated work.

## Workspace and deliverables

Resolve the sibling path relative to the actual MD-tools checkout, not a hard-coded machine path.
If it exists, inspect and preserve its contents rather than overwriting it.

Keep the project independent of the engine:
- inputs/: chemical identity, stereochemical SMILES, source/reference and atom mapping.
- config/: build, REST2, CV and relevant preparation configurations.
- scripts/: small reproducible preparation, run and validation scripts.
- data/: generated build and simulation datasets; ignored by Git.
- reports/: measured checks, commands, plots and final verdict.
- README.md: short reproduce/resume/register instructions and engine pin.

Version the small project inputs/configs/scripts/reports locally; do not commit trajectories,
checkpoints, charge caches or large generated data. Do not create a remote sibling repository
without a request. At the end, push a concise, self-contained integration report to MD-tools dev
at docs/integration/rgdfv-rest2.md, including the validated molecular identity, essential settings,
commands/results, actual engine pin and registered dataset identity. Engine fixes, if needed,
belong in MD-tools, not in copied runtime code inside the project.

## 1. Establish the molecule before building

Target head-to-tail cyclo(L-Arg-Gly-L-Asp-D-Phe-L-Val), c(RGDfV).
Lowercase f means D-Phe. This is the non-N-methylated molecule; do not substitute a similarly
named N-methylated analog.

Use the user's existing trusted input if available. Otherwise construct/obtain a chemically
explicit stereochemical SMILES and validate it independently before use. Record its origin.
Specify Arg guanidinium +1 and Asp side-chain carboxylate -1, for net formal charge zero.
There are no free backbone termini. Confirm:
- one connected cyclic pentapeptide with a 15-membered backbone ring;
- exactly five backbone carbonyl-C--amide-N links, including Val->Arg ring closure;
- L-Arg, L-Asp, D-Phe and L-Val stereochemistry, with Gly achiral;
- explicit protonation, hydrogens and all bond orders;
- source graph, prepared SDF and built topology agree through an explicit atom mapping.

Use residue chemistry and stereocentres rather than copying ALA atom indices.
Create a residue/backbone atom map from the molecular graph, retained across conversion.
Generate an annotated structure or atom table sufficient to review the five omega bonds and CVs.

## 2. Parameterize through the Sage whole-molecule route

Use the public build-top interface and a .smi input. Essential configuration:

solute:
  peptide: false
  ligand_forcefield: sage-2.2.1
  ligand_charge_method: am1bcc
solvent:
  model: GBn2
hydrogen_mass_repartitioning:
  enabled: false

Use HBonds constraints and a 2 fs timestep for dynamics. No periodic box, explicit water,
ions, cutoff, barostat or NPT stage. No SASA/nonpolar term for this first test.
Inspect the installed configuration schema for exact supported fields; do not invent flags.

The peptide-PDB route uses a protein force field and would not test Sage. The actual intramolecular
force field must be openff-2.2.1, with the exact resource/version recorded.

### Charges

Require actual AM1-BCC, not NAGL, Gasteiger or a silent fallback. The current code may choose
am1bccelf10 when OpenEye is detected even when am1bcc was requested. Inspect the actual call and
record the actual toolkit backend/charge scheme. For this experiment use the AmberTools AM1-BCC
route; an ELF10 run does not satisfy this particular test.

Record AmberTools/OpenFF/toolkit versions, conformer preparation and any charge cache used.
Retain per-atom charges with the atom mapping and the charge calculation diagnostics. Verify
finite charges, net charge agreement within 1e-5 e, complete parameter assignment, correct
particle/bond counts, preserved stereochemistry and ring closure.

### Implicit solvent is a separate component

Inspect the final System's GBn2 parameters and radii. Sage controls the intramolecular parameters;
the GB model/radius assignment is an additional modelling choice.

In particular, the single-ligand residue representation may prevent mbondi3's residue-name-based
Arg/Asp corrections. Report what was actually assigned to the Arg/Asp functional groups and
whether the requested radius change was a no-op. Do not relabel residues or invent radii merely
to make a check pass.

A documented generic ligand-radius assignment may be used as an explicitly labelled exploratory
Sage+GBn2 integration test. Do not claim conventional peptide mbondi3 parity if those adjustments
were not applied. If the resulting Hamiltonian is internally inconsistent or a required parameter
is missing, fix or report that before running; a complete but scientifically unvalidated hybrid
model is a limitation, not proof of invalid software.

Retain SDF, charge table, parameterization provenance and serialized System in the eventual dataset.
Verify the conversion to the final implicit System preserves the intended charges and bonded/
nonbonded parameters, apart from the explicitly added GB force and configured constraints.

## 3. Prove omega exclusion at the force level

Identify the five backbone amide central bonds independently from the source chemical graph.
Use the actual automatic ligand classifier first; do not pre-populate its expected output.

Pass if:
- the resolved exclusion set contains exactly those five ordinary backbone amide bonds;
- the ring-closing bond is included;
- no backbone nitrogen is misclassified as proline-like because it lies in the macrocycle;
- no candidate remains ambiguous and no unrelated central bond is excluded;
- the SDF-to-topology index mapping is demonstrated.

Inventory every PeriodicTorsionForce term about each excluded central bond, including all
periodicities and reversed atom order. Compare the original and scaled Systems at tau=0, 0.25, 0.5.
For all those terms, periodicity, phase and force constant remain unchanged. The torsion is
retained, not deleted, restrained, or removed from the force field.

Positive control: nonzero eligible non-omega torsion terms scale by (1-tau)^2. Also check the
existing intended scaling for solute nonbonded/1-4 and GB, and that bonds/angles remain unchanged.
At tau=0 the scaled System reproduces the original Hamiltonian. Reuse established repository
parameter/energy comparison tolerances and record them before inspecting results.

Save a compact bond/term table with expected versus observed factors. Omega trajectories are a
useful diagnostic but are not the proof: an unscaled omega may still fluctuate or isomerize.

If detection is wrong, fix the reusable detector/mapping in MD-tools with a focused regression
test. An explicit manual exclusion can help diagnose the issue, but cannot be reported as passing
automatic detection.

## 4. Run the integration experiment

Use the accepted protocol:
- six states: tau = 0, 0.1, 0.2, 0.3, 0.4, 0.5;
- one physical temperature, 300 K;
- 2 fs timestep, HMR off, fixed-volume/nonperiodic implicit solvent;
- minimize, then 100 ps equilibration at each state's Hamiltonian;
- 1 ns REST2 production per state;
- exchanges every 10 ps;
- CVs every 1 ps; state trajectories and checkpoints every 10 ps.

At 2 fs this is 50,000 equilibration steps per state, 500,000 production steps,
5,000 steps per exchange and 500 steps per CV observation.
Use configured Langevin settings explicitly and record deterministic seeds.
Run through supported public interfaces/installed APIs; inspect how per-state initialization is
supported before assembling the equilibration pipeline. Do not forge checkpoint files or
silently replace per-state equilibration with six copies of one tau=0 state.

Use real CUDA and six MPI ranks for the ladder. Inspect available devices, prefer one rank per
available GPU, and use only documented device sharing if necessary. Record rank/device mapping.
Do not occupy or terminate unrelated jobs. AM1-BCC and graph checks legitimately run on CPU.

Define all 15 backbone CVs (five phi, five psi, five omega) with cyclic indexing.
Validate definitions against the mapped chemical graph and independently recompute selected
reported rows from matching saved coordinates. Respect the documented pre-exchange CV convention:
do not compare a pre-exchange row to a post-exchange frame holding another configuration.

Deliberately interrupt once after at least 200 ps of production, retain the committed checkpoint,
then resume to the original 1 ns budget. Do not extend the budget accidentally.
Verify checkpoint progress, expected CV grids, trajectory/state assignment and no lost/duplicated
committed observations. Report exchange attempts/acceptance, state/walker movement, energies,
temperatures, ring integrity and omega traces. Acceptance and sampling quality are observations,
not targets to force by changing the agreed ladder mid-run.

This budget establishes integration behavior, not converged populations or a validated force field.

## 5. Finish, register and verify

Keep build, preparation and production provenance together in a self-contained dataset under
the sibling project's data/. Include the molecular identity/protonation, mapped charges,
force-field/GB details, actual engine pin, configs/seeds, resolved exclusions, all CV definitions
and outputs, per-state trajectories, exchange history, checkpoints and completion records.
Include essential validation reports before registration; registered datasets remain immutable.

Resolve storage through the existing machine/user configuration, MD_DATA or documented override.
Never invent a storage root or overwrite user configuration. If none is available, finish the
local experiment and report the missing destination as the deposition blocker.

Use the current data-contract v2 interface:
md-openmm data-register -idata <finished-dataset> -project_name RGDfV-REST2 -data_name RGDfV-Sage221-GBn2-REST2 -year <completion-year> --dry-run
then the same command without --dry-run, followed by:
md-openmm data-register -idata <registered-destination> --verify-only

The expected project-role destination is:
$MD_DATA/<completion-year>/RGDfV-REST2/RGDfV-Sage221-GBn2-REST2/

The user has authorized this integration/deposition experiment. After validation and a successful
dry run, execute registration using the supported transaction. Never manually move/delete the
source to simulate successful registration. If the destination already exists with different
content, use a clearly versioned new dataset name and record it; never overwrite it.

Pass if completion/lineage checks and dry run succeed, real registration succeeds, verification
passes, the original data path becomes the intended symlink, and analysis can reopen trajectories/
CVs through that link. Confirm the dataset contains the inputs and metadata needed to interpret
the output without referring to temporary build directories.

Registration into local managed storage tests the FAIR-oriented contract, not public FAIR
certification. Record dataset identity, checksums, creator, method/software provenance, access
location and reuse terms where known. Do not invent a license or claim a DOI/public deposit.

## 6. Fix only blockers, then hand off

Start with parameterization and force-level checks; do not spend the REST2 budget on a wrong
molecule or incorrect scaling. If a check fails, preserve the command/diagnostic, fix the concrete
cause, and rerun the affected check. Do not loosen scientific assertions or substitute another
force field/charge model.

If the engine needs a fix, implement it on MD-tools dev, add a focused regression test, run the
affected existing CUDA/MPI lanes, install the new commit into the sibling environment, and record
the changed pin. Rerun this experiment's dependent checks; run a full suite only if shared
scientific execution or restart logic changed. Do not reopen deferred metadata hardening.

Keep unrelated optional issues in the backlog. Stop on a genuine chemical/parameterization/
hardware blocker with the finished work and diagnostic intact; never label an unexecuted step
as passing.

Final report:
- molecule, stereochemistry/protonation and actual parameter/charge/GB choices;
- five-bond omega exclusion table and positive scaling controls;
- measured six-state run and interrupted-resume results;
- actual engine commit and any focused fixes;
- registration/verification result, dataset ID and actual destination;
- concise limitations and next scientific step.

Push the integration report and any necessary engine fixes to MD-tools dev. Keep the project
files in the sibling project and large data in the registered dataset. Do not start a longer
production study automatically.
