# Evidence-based OpenMM defaults and scientific rationale

## Purpose

Update the current simplified OpenMM template so its defaults are conservative for method
development, its alternative choices remain available, and every consequential choice is justified
by traceable evidence.

This task follows the current repository architecture. Do not restore a profile registry, schema
migration layer, workflow manager, legacy root-level generators, or any other deleted framework.
Keep the six public commands and the canonical generators:

- `src/md_templates/openmm/sysgen.py::generate_system()`
- `src/md_templates/openmm/mdgen.py::generate_md()`

Work on `dev`. Do not merge to `main`, create a release, or move a tag. The repository is already
at `0.4.0.dev0`; do not relabel the meaning of data produced by 0.3.x.

Use the Claude Code plugin's `academic-research` skill for the literature work described below.
Invoke that skill before drafting the scientific rationale. Do not invent citations if the skill or
a primary source is unavailable.

## 1. Audit before changing defaults

Read the current implementation, generated-script templates, tests, README, support matrix,
`CLAUDE.md`, and the latest journal entries. Trace each public value from:

`show-default/sys-config -> resolved YAML -> builder -> serialized OpenMM System -> provenance`.

Record the audit in:

`docs/journal/2026-08-27_evidence-based-defaults-and-rationale.md`

The journal must identify all current duplicate or stale declarations, including
`defaults.py`, `builder_defaults.py`, builder fallbacks, examples, and tests. Change the canonical
public default and every still-active consumer. Do not add another copy of the defaults.

## 2. Explicit-solvent default and alternative

Change the default explicit-solvent combination to:

| component | required default |
|---|---|
| protein/peptide | AMBER ff14SB |
| small-molecule ligand | OpenFF Sage 2.2.1 |
| ligand charges | standard AM1-BCC through the recorded toolkit route |
| water | TIP3P |
| ions | 0.15 M NaCl, with neutralising ions recorded separately |
| nonbonded treatment | PME, 1.0 nm real-space cutoff |

Pin the ligand force field by its canonical installed identifier, expected to be
`openff-2.2.1`/`openff-2.2.1.offxml`. Verify the exact resource name in the installed environment
rather than guessing.

Keep the existing ff19SB + OPC choice as an explicit alternative:

| component | optional combination |
|---|---|
| protein/peptide | AMBER ff19SB |
| small-molecule ligand | OpenFF Sage 2.2.1 |
| water | OPC |

Do not create a profile registry or inheritance system. Extend the existing simple solvent
selection:

- `md-openmm sys-config ... --solvent TIP3P` selects ff14SB + TIP3P and is the explicit default.
- `md-openmm sys-config ... --solvent OPC` selects ff19SB + OPC.
- `md-openmm sys-config ... --solvent GBn2` retains the implicit route below.

Configuration remains ordinary editable YAML. Use canonical, loadable OpenMM/OpenMMForceFields
resource names consistently. Reject a resource that cannot actually be loaded. Do not advertise a
combination solely because its strings look correct.

For protein-only systems, do not claim that Sage participates. For ligand-containing systems,
record the exact ligand force field and charge route. Do not silently substitute NAGL for standard
AM1-BCC.

## 3. Implicit-solvent default and scope

Retain the peptide/protein implicit default:

- ff14SB;
- GBn2;
- mbondi3 radii;
- no SASA/nonpolar surface-area term;
- nonperiodic system;
- no pressure and no barostat.

Describe this as matching the intended Amber `igb=8`, `mbondi3`, `gbsa=0` model only where the
implemented radii and force terms actually support that statement.

Audit the nonpeptidic Sage + GBn2 route carefully. Show how every ligand atom receives a GB radius
and parameter. If arbitrary Sage ligands have not been validated against mbondi3/GBn2, label that
route experimental in user-facing documentation and provenance, or fail clearly for unsupported
chemistry. Do not claim exact Amber `igb=8` parity for an arbitrary OpenFF ligand without
numerical and parameter-assignment evidence.

## 4. Box and nonbonded defaults

Change the explicit dodecahedral padding default from 2.0 nm to 1.5 nm. Keep 2.0 nm as a
user-selectable conservative override for unusually flexible solutes, unfolded states, or enhanced
sampling expected to expand the solute.

Retain:

- dodecahedral box;
- PME;
- 1.0 nm real-space cutoff;
- the existing minimum-image/cutoff-fit safety gate;
- the existing deterministic solvation and explicit salt/counterion accounting.

The documentation must explain OpenMM's actual padding semantics and distinguish:

1. requested solute-to-box clearance;
2. solute-to-periodic-copy clearance;
3. shortest periodic box height;
4. the cutoff plus minimum-image margin.

Do not claim that 1.5 nm universally prevents self-interaction for every future conformation. The
built-system gate must verify that the actual box is compatible with the cutoff, and the rationale
must explain when 2.0 nm is appropriate.

Record in resolved configuration and provenance:

- nonbonded method;
- cutoff;
- switching-function state and switch distance;
- dispersion-correction state;
- Ewald error tolerance;
- box vectors and volume;
- requested and resolved padding/clearance information.

## 5. Thermostat default

Retain the OpenMM `LangevinMiddleIntegrator` default with:

- temperature 300 K;
- friction coefficient `1.0 ps^-1`;
- conservative default timestep 2 fs.

Use OpenMM language in configuration: `friction_per_ps`, not an ambiguous prose-only “collision
time”. The rationale may state that `1.0 ps^-1` corresponds to a nominal 1 ps damping/collision
time.

Explain the distinction between equilibrium configurational sampling and real-time dynamical or
transport observables. Do not claim that thermostat choice is irrelevant to kinetics.

## 6. Monte Carlo barostat

Expose and implement one public barostat frequency, in steps, with the OpenMM default of:

`barostat_frequency_steps: 25`

The generated standalone scripts, resolved stage files, and provenance must use and record this
value. Explain that it corresponds to 0.05 ps at 2 fs and 0.10 ps with the optional 4 fs timestep.

Preserve the scientific invariants:

- minimization: barostat disabled;
- NVT: barostat disabled;
- explicit NPT: exactly one active `MonteCarloBarostat`;
- implicit solvent: no barostat in the System;
- each REST2 replica: a distinct deterministic barostat seed.

Do not confuse the presence of a barostat Force with whether it is active in a stage.

## 7. HMR plus 4 fs is an option, not the baseline

Keep the conservative default:

- `hydrogen_mass_amu: null`;
- `timestep_fs: 2.0`.

Support and document an explicit performance option:

- HMR target hydrogen mass `3.024 amu`;
- `timestep_fs: 4.0`;
- hydrogen-involving bonds constrained;
- rigid water for explicit solvent.

Add a short editable YAML example. Reject 4 fs without HMR and the required constraints. Verify
that applying HMR conserves each bonded heavy-atom/hydrogen group's total mass and does not produce
nonpositive heavy-atom masses.

The rationale must distinguish:

- evidence that HMR can permit stable 4 fs integration in suitable biomolecular systems;
- equilibrium configurational sampling;
- altered masses, time correlation functions, kinetic rates, diffusion, and transport properties;
- the need for a short energy/stability check for a new system class.

Never present HMR + 4 fs as validated for quantitative kinetics merely because a short smoke run is
stable.

## 8. FAIR-ready force-field and protocol record

Ensure generated records contain what was actually built, not only what was requested:

- MD-templates version and commit SHA;
- OpenMM, OpenFF Toolkit, openmmforcefields, AmberTools and relevant toolkit versions;
- exact force-field resource identifiers;
- protein, ligand, charge and water choices;
- explicit/implicit route and GB radii/SASA state;
- PME, cutoff, switching, dispersion and Ewald settings;
- box shape, vectors, volume, padding and salt accounting;
- constraints, rigid-water setting, hydrogen masses and timestep;
- thermostat and friction;
- pressure, barostat frequency and seeds;
- whether a value was requested, resolved, derived or unavailable.

Do not retrofit the new default onto 0.3.x data. The retrospective tool must preserve the original
ff19SB/OPC identity when recorded. Missing historical values remain `unknown` or explicitly
`inferred` with evidence; they must never inherit the current default.

## 9. Academic research and PDF deliverables

Use the Claude Code plugin's `academic-research` skill to create all three:

- `docs/md-defaults-scientific-rationale.md`
- `docs/md-defaults-scientific-rationale.pdf`
- `docs/md-defaults-references.bib`

Keep the main rationale focused, approximately 6–10 pages excluding references. It must justify:

1. ff14SB + Sage 2.2.1 + TIP3P as the method-development explicit default;
2. ff19SB + Sage 2.2.1 + OPC as an available alternative, without claiming joint optimization;
3. ff14SB + GBn2 + mbondi3 without SASA for peptide/protein implicit solvent;
4. the limitation of GBn2/mbondi3 for arbitrary nonpeptidic Sage ligands;
5. a dodecahedron with 1.5 nm default padding and a 2.0 nm conservative option;
6. PME and the 1.0 nm cutoff;
7. Langevin-middle with `1.0 ps^-1` friction;
8. the Monte Carlo barostat and 25-step attempt frequency;
9. 2 fs without HMR as the baseline;
10. HMR + 4 fs as an optional efficiency setting and its kinetic limitations.

For every consequential claim, classify the evidence as one of:

- jointly parameterized or directly developed;
- directly benchmarked as a complete combination;
- component-level evidence;
- scientifically reasonable extrapolation;
- implementation documentation.

Correctly state that the original Sage 2.0 protein–ligand benchmark used ff99SB*-ILDN with TIP3P,
not ff14SB or ff19SB. Separately discuss later direct workflow-level evidence using ff14SB,
Sage 2.2, and TIP3P. Do not write that Sage was fitted to protein-binding affinities, and do not
write that all Sage parameters were fitted against TIP3P.

Prefer original peer-reviewed papers for ff14SB, ff19SB, TIP3P, OPC, Sage, GBn2, PME, Langevin-middle
integration, Monte Carlo pressure coupling, and HMR. Use official OpenMM/OpenFF documentation only
for current implementation semantics. Verify every title, author list, year, journal, DOI, and URL.
Do not cite a search-result snippet or secondary summary when the primary source is available.

Include:

- a table of every default, alternative, evidence level, source, and main limitation;
- a section called “What these citations do not establish”;
- a concise recommendation for method-development comparisons;
- a reproducibility note connecting the paper's choices to generated provenance.

Keep the editable Markdown and BibTeX authoritative. Generate the PDF from them using the simplest
available reproducible route; do not introduce a documentation framework. Record the exact render
command and tool versions in the journal. Render the PDF to images, inspect every page, and correct
overflow, missing glyphs, broken references, blank pages, and unreadable tables. Verify that DOI
and documentation links resolve.

Link both the Markdown and PDF from the root README. The implemented defaults, README summary,
support matrix, Markdown rationale, PDF tables, examples, and provenance names must agree exactly.

## 10. Focused tests only

Do not repeat nanosecond validation or construct a combinatorial test matrix. Add the smallest tests
that would have caught these changes being incomplete:

1. `show-default sys` and `sys-config` default to TIP3P/ff14SB/Sage 2.2.1.
2. `--solvent OPC` resolves to ff19SB/OPC and remains buildable.
3. `--solvent GBn2` resolves to ff14SB/GBn2/mbondi3/no-SASA and never adds a barostat.
4. The exact force-field resources load in the installed environment.
5. Generated provenance reports the built force fields, water, charge route and nonbonded settings.
6. The 1.5 nm default reaches the built-system cutoff/minimum-image gate.
7. The public barostat frequency reaches every explicit NPT/REST2 barostat and is inactive in NVT.
8. The default generated integrator is 2 fs without HMR.
9. The 3.024 amu HMR + 4 fs option changes masses safely and generates a 4 fs integrator.
10. Unsafe 4 fs configurations fail with a specific message.
11. Existing 0.3.x retrofit fixtures do not acquire the new defaults.

Any test that minimizes or integrates a molecular system must run on CUDA and carry the `gpu`
marker. Use picosecond smoke sizes only. Non-GPU tests may inspect configuration, generated source,
serialized metadata, or pure functions, but must not perform MD. The GPU-less release workflow
continues to be packaging-only.

Run fast non-MD tests during development, then one short CUDA acceptance pass covering the new
default and the optional HMR path. Do not rerun the earlier long cMD or REST2 campaigns for this
defaults/documentation task.

## 11. Documentation and completion

Update:

- `README.md`;
- `CLAUDE.md` where durable defaults or safety rules changed;
- `docs/support-matrix.md`;
- active YAML examples and CLI help;
- `claudecode-instructions/README.md`, making this the current task;
- the journal named above.

Remove or correct active statements that still call ff19SB/OPC or 2.0 nm padding the default.
Historical instructions and journals may retain their original statements for provenance; do not
rewrite history.

Before completion:

1. inspect the full diff for duplicate default declarations and unsupported claims;
2. run the focused fast suite;
3. build/load one ff14SB/Sage 2.2.1/TIP3P system;
4. build/load the ff19SB/Sage 2.2.1/OPC alternative;
5. perform only the short CUDA runtime checks specified above;
6. render and visually inspect the PDF;
7. verify citations and links;
8. commit and push to `dev`;
9. do not merge or tag.

The final report must give:

- commit SHA;
- exact files and defaults changed;
- exact tests and results, separating non-MD from CUDA MD checks;
- the OpenMM platform and GPU used for every MD check;
- exact resolved force-field resource names;
- evidence that TIP3P is now the default and OPC remains selectable;
- implicit-solvent scope and remaining ligand limitation;
- box-clearance and cutoff-gate evidence;
- thermostat, barostat and HMR evidence;
- PDF path, page count, render/inspection method and citation count;
- any claims downgraded from direct evidence to extrapolation;
- remaining scientific and implementation limitations.

Do not return PASS merely because the documentation is polished. Conversely, do not launch long
production simulations: this task accepts short CUDA execution plus direct inspection of the built
Systems and provenance.
