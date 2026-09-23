# 0.7.0 — conventional alchemy: TI and FEP

**Under construction.** Nothing here is in a release, and no part of it is available in the
installed package. 0.6.0 is still under the user's testing; 0.7.0 is released only after 0.6.1 is
released and integrated.

Baseline: `e524e0e` on the 0.6.0 development line (released as `v0.6.0` at `3927105`; the `dev-0.6.0` branch has since been deleted).

## Purpose

A reproducible path from two registered ligand parameter packages to a free energy with an
uncertainty: absolute and relative solvation free energies, absolute and relative binding free
energies, computed by thermodynamic integration and by free-energy perturbation over one defined
Hamiltonian.

This is deliberately conventional. The package already has a two-System mixing machinery in
`md_tools.ais.two_state`, but it is linear in `lambda` between two Systems with **identical
particles** (`_structural_differences` refuses anything else). Alchemy needs a mapped topology and
a softcore potential; it is a different Hamiltonian with a different derivative, and it must not
be built by relabelling the AIS one.

## User-visible outcome

A new subcommand under the existing single executable:

~~~text
md-openmm combine-topology   two endpoint parameter sets + a map -> a combined topology plan
~~~

This is the one authorized exception to the four-command rule in `CLAUDE.md`, granted explicitly
for this release. It stays a subcommand of `md-openmm`; there is no second executable.

Then lambda-window execution through the existing runtime, and analysis producing a free energy
with an uncertainty and overlap diagnostics.

## Terminology, defined by representation

These terms are defined by **particle representation and interactions**, not by how many files are
involved:

- **Single topology** — one evolving atom representation, with explicit handling of unmatched or
  dummy particles where supported.
- **Dual topology** — distinct endpoint ligand particle sets present in one environment, with
  explicit mutual exclusions and a stated positional/restraint treatment.
- **Hybrid topology** — a mapped shared core plus separate endpoint-unique particles.
- **Separated topology** — **deferred**. Not implemented, not advertised, not partially supported.

OpenFE's hybrid machinery alone does not establish support for all three of the first three modes.
Each is supported when its own construction checks and endpoint recovery pass.

## Milestones

| | milestone | content |
|---|---|---|
| A0 | interfaces | written interfaces, input/output schemas, fixed reference fixtures, OpenFE adapter choice |
| A1 | `combine-topology` | explicit maps, combined coordinates, endpoint records, single/dual/hybrid construction checks |
| A2 | Hamiltonian | validated Amber18 softcore, complete state energies and derivatives |
| A3 | windows and estimators | window execution, FEP/BAR/MBAR and TI, absolute and relative hydration examples |
| A4 | relative binding | solvent and complex legs, a validated simple ligand pair |
| A5 | absolute binding | restraints, restraint free energies, standard-state correction |
| A6 | integration | CUDA/CI, exports, registration, documentation, installed-wheel validation |

These are staged, not a licence to drop binding or topology modes from the aims. Until a
combination is validated it is marked unsupported and refused; a successful short run does not
create support.

## combine-topology

Construction is independent of sampling and estimation: it takes registered endpoint parameters
and an explicit — or a validated automatic — atom map, and produces a reviewable plan.

- Parameter values and chemical states are **immutable**. A standard OpenFE workflow must never
  silently reparameterize a registered MD-tools ligand; the package's parameters are what the
  package says they are, and a run that changed them is a different experiment.
- One matched environment. Two independently solvated endpoint boxes are not automatically
  atom-matched, and treating them as if they were is a silent error.
- The plan records: endpoint package and System identities, atom maps in **both** directions,
  common / A-only / B-only particle sets, environment identity, coordinates, constraints, masses,
  virtual sites, force-field conventions, exclusions, and the hashes of every source.
- Validated: mapping symmetry, chemical ambiguity, force-field compatibility, missing terms,
  duplicate or omitted interactions, and endpoint recovery.
- Dummy partition-function contributions, retained bonded terms and restraint effects must cancel
  by a demonstrated construction, or be corrected explicitly. **Comparing raw endpoint energies
  without accounting for dummy terms is not a valid endpoint test** — it can pass while the
  construction is wrong.

The CLI wiring lives in the existing `src/md_tools/cli/md_openmm.py`, reusing the existing builder,
configuration and preflight authorities. No second MD runner, no second parameter catalog.

## Amber18 softcore, on by default

~~~yaml
alchemical:
  sc: true
  softcore_function: amber18
  scalpha: 0.5
  scbeta: 12.0   # angstrom^2; converted internally to 0.12 nm^2
~~~

These are **MD-tools defaults**, chosen by the user. They are not a claim that AMBER's native
`ifsc` defaults to 1.

Implement the Amber18 LJ and electrostatic softcore definitions of the Amber18 manual section
21.1.5, equations 21.5–21.7, with R4 as the implementation reference. Define the appearing and
disappearing directions explicitly, and every intramolecular and exception rule.

Two traps, named because they are easy to fall into and invisible afterwards:

- OpenFE's default Gapsys potential, and its Beutler LJ option, are **not** Amber18 equivalence.
  Selecting one and calling it `amber18` would be a false label on a validated-looking number.
- Electrostatic softening must be validated with the correct PME decomposition: reciprocal space,
  self and background terms where applicable, exclusions, and dispersion corrections. Softening a
  real-space Coulomb term while its original contribution survives somewhere else is the classic
  failure, and it produces smooth, plausible, wrong free energies.

The softcore functional form is separate from the lambda schedule. Record whether electrostatics
and sterics change together or in stages, validate the chosen schedule and explain it. A later
AMBER smoothstep variant is a different function and must not ship under the name `amber18`.
`sc: false` selects a documented ordinary path for cases that support it, and refuses unsafe
particle insertion or removal rather than quietly re-enabling softcore.

## TI and FEP are analyses of one Hamiltonian

Write cross-state reduced potentials with explicit sample and state IDs. FEP, BAR and MBAR consume
them; TI consumes derivatives from the same path. They are not independent implementations of the
potential — if they were, agreement between them would prove nothing.

For TI, report the complete derivative along path progress `s`:

~~~text
dU/ds = sum_k (partial U / partial lambda_k) (d lambda_k / ds)
~~~

including softcore, electrostatics and PME, exception, bonded, restraint and applicable long-range
correction terms. **OpenMM custom-force derivatives alone do not prove complete coverage of
`NonbondedForce` parameter offsets.** Use justified analytic derivatives or controlled numerical
ones, with documented error and finite-difference checks over several step sizes — a solvent
background energy of tens of thousands of kJ/mol will hide a missing term otherwise.

Use stable exponential averaging for FEP and report overlap limitations honestly.

**Do not reuse the AIS identity `dU/dlambda = U1 - U0`.** It is exact for AIS because that path is
linear; a softcore path is not, and carrying the identity across would silently give wrong
derivatives. The existing AIS mode keeps its contract unchanged. A future alchemical AIS needs its
own defined contract, and if switching work is used there, work remains the frozen-coordinate
energy difference.

## Execution

Fixed-lambda windows first. Exchange between windows, if added, goes through the existing MPI and
exchange authorities — not a second implementation. Preserve LangevinMiddle, the established
timestep and HMR validation, truthful CUDA selection, transactional resume, and the read-only
failed preflight.

Alchemical NPT needs ensemble-correct state energies and checkpoints; REST2's NVT rule belongs to
REST2 and is not imposed on ordinary alchemy.

## Thermodynamic cycles

- **Absolute binding**: the full cycle, Boresch-style or otherwise validated restraints, the
  restraint attachment and release contributions, and the standard-state correction. A binding
  free energy missing the standard-state term is a different quantity wearing the same name.
- **Relative hydration**: vacuum and solvent legs.
- **Relative binding**: solvent and complex legs, with consistent orientation and sign conventions.

## Supported chemistry, initially

Start with neutral ligands and charge-preserving, simple substitutions. Before broadening, write
down the validation each of these needs: net-charge changes and their finite-size corrections,
ring changes, stereochemistry, constraints, virtual sites, covalent ligands, metals. Until then
each is refused by name. A successful short run does not establish support for a chemistry class.

## Non-goals

- Separated topology.
- Any change to the existing cMD, REST2 or AIS contracts.
- Alchemical AIS.
- Making alchemy mandatory: it stays optional for existing cMD installations.

## OpenFE reuse

Use OpenFE as the implementation reference for mapping and hybrid construction, schedules,
cross-state energies, multistate sampling, and BAR/MBAR-compatible analysis. Prefer a thin adapter
around pinned components over copying the project. Any vendored code is documented with its
license, its commit, and every deliberate difference. Reviewed source commit for R6:
`446961f654e2f2ba9145d5b8de1ce90ef41122ff`; its documented dummy-group limitation must be
addressed by the chosen construction, not inherited.

## References

R4 Lee et al., Amber18 (<https://doi.org/10.1021/acs.jcim.8b00462>) · R5 Amber18 manual section
21.1.5 (<https://ambermd.org/doc12/Amber18.pdf>) · R6 OpenFE relative hybrid protocol ·
R7 OpenFE absolute protocols · R8 PyMBAR (<https://pymbar.readthedocs.io/>) · R9 OpenMM
`NonbondedForce`. Full register: section 10 of the
[20260918 instruction](../../claudecode-instructions/20260918_parallel-0.6.1-0.7.x-development.md).
Pin exact versions and commits when implementation of each part begins.

## Acceptance criteria

Written into an acceptance matrix before any long campaign; each row names a fixture, the expected
quantity, an independent reference, the tolerance, the platform, the command and the evidence
path. See [validation gates](../shared-contracts.md#validation-gates).

The free-energy agreement gate for matched simple references: the absolute discrepancy is below
**both** 0.5 kcal/mol **and** three combined standard errors, including the integration and
reference uncertainty, with a justified numerical floor for effectively exact fixtures.
Inconclusive uncertainty is not a pass. A tolerance is never weakened after a failure.
