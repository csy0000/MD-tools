# Scientific rationale for the MD-templates OpenMM defaults

| | |
|---|---|
| Repository | MD-templates (`csy0000/MD-templates`), version 0.4.0.dev0 |
| Applies to | the values `md-openmm sys-config` and `md-openmm show-default` write |
| Date | 2026-08-27 |
| References | `docs/md-defaults-references.bib` |
| Rendered by | `python scripts/render_markdown_pdf.py docs/md-defaults-scientific-rationale.md docs/md-defaults-scientific-rationale.pdf` |

---

## 1. What this document is, and what it is not

This document states, for each consequential default in the OpenMM workflows this repository
generates, what the choice is, what evidence supports it, and what that evidence does **not**
support. It exists because a default is a scientific claim made on behalf of every user who does
not change it, and because the alternative — a table of values with no provenance — makes the
claim silently.

It is not a validation report. This repository has run no production campaign to justify these
values; the evidence below is the published literature plus what can be read off the systems the
code actually builds. Section 12 lists what remains unestablished.

**Evidence classification.** Every claim in the tables below carries one of five labels. They are
ordered by strength and they are not interchangeable:

| label | meaning |
|---|---|
| **JP** — jointly parameterized | the components were developed together, or one was fit in the presence of the other |
| **CB** — combination benchmarked | the complete combination was benchmarked against experiment or reference data in a published study |
| **CE** — component evidence | each component is supported individually; the combination is not itself the subject of a study |
| **EX** — extrapolation | a scientifically reasonable inference from adjacent evidence, stated as an inference |
| **ID** — implementation documentation | a fact about what the software does, not about whether it is right |

An **ID** source is never used in place of a primary source for a scientific claim. Where a value
is taken from software documentation because it *is* a software fact — OpenMM's barostat default
frequency, for instance — that is said plainly.

---

## 2. The defaults, in one table

| setting | default | alternative | evidence | limitation |
|---|---|---|---|---|
| protein force field | ff14SB (`amber14-all.xml` → `amber14/protein.ff14SB.xml`) | ff19SB (`amber19-all.xml`) | **JP** with TIP3P (§3) | backbone correction is empirical and TIP3P-specific |
| water | TIP3P (`amber14/tip3p.xml`) | OPC (`amber19/opc.xml`) | **JP** (§3) | TIP3P misrepresents bulk water; ff14SB relies on error cancellation with it |
| ligand force field | OpenFF Sage 2.2.1 (`openff-2.2.1`) | — | **CE** + **CB** at Sage 2.0 (§4) | 2.2.1 itself has no published protein–ligand benchmark |
| ligand charges | AM1-BCC via AmberTools `sqm` | `am1bcc_nagl` (explicit) | **JP** with Sage (§4) | NAGL predicts AM1-BCC ELF10, it does not compute it |
| implicit solvent | GBn2 + mbondi3, no SASA | — | **JP** with ff99SB/ff14SB (§5) | peptides and proteins only; see §6 |
| box | rhombic dodecahedron | — | **CE** (§7) | — |
| padding | 1.5 nm | 2.0 nm | **CB** (§7) | not a guarantee for any future conformation |
| electrostatics | PME, 1.0 nm real-space cutoff | — | **CE** + **ID** (§8) | finite-size artifacts are reduced, not removed |
| thermostat | `LangevinMiddleIntegrator`, 300 K | — | **CE** (§9) | — |
| friction | 1.0 ps⁻¹ | — | **CE** + **EX** (§9) | affects kinetics and transport, not just sampling speed |
| barostat | `MonteCarloBarostat`, 1 bar | — | **CE** (§10) | — |
| barostat frequency | 25 steps | — | **ID** + **CE** (§10) | 25 is OpenMM's default, not an optimum for any system |
| timestep | 2 fs | 4 fs with HMR | **CE** (§11) | — |
| hydrogen mass | unmodified | 3.024 amu | **CB** for stability (§11) | dynamics, kinetics and transport are altered |
| ionic strength | 0.15 M NaCl, counterions separate | — | **CE** | ion parameters are whatever the water XML carries |

---

## 3. ff14SB + TIP3P as the explicit default, ff19SB + OPC as the alternative

### 3.1 The two pairs are each internally consistent, and they are not interchangeable

ff14SB's side-chain dihedrals were fit to gas-phase QM (`HF/6-31G*` geometries, `MP2/6-31+G**`
single points), but its backbone adjustment was not: it is, in the authors' words, "an empirical
correction based on data obtained from simulations carried out in TIP3P explicit water"
[@maier2015ff14sb]. Every validation simulation in that paper used TIP3P, and the paper closes the
point with an explicit caution — "transferability of this correction term to other implicit or
explicit solvent models may need evaluation prior to production use."

ff19SB replaced that empirical backbone correction with amino-acid-specific CMAP maps trained
against QM in aqueous (implicit) solvent, and its authors recommend a different water model:
"Of the explicit water models tested here, we recommend use of OPC with ff19SB" [@tian2020ff19sb].
The same paper is blunt about why the older pair works: ff14SB has an inherent underestimation of
helicity that TIP3P's compact-structure bias partly cancels.

So the two pairs are not two spellings of the same thing. **ff14SB + TIP3P** is a pair whose known
error cancellation is the reason it reproduces the benchmarks it reproduces; **ff19SB + OPC** is a
pair whose backbone was fit to be right for a better water model. Mixing them across — ff19SB with
TIP3P, or ff14SB with OPC — discards the reason either works. This repository therefore ships the
protein force field and the water model as **one selection**, not as two independent keys with
independent defaults.

*Evidence: **JP** for each pair. [@maier2015ff14sb; @tian2020ff19sb; @jorgensen1983tip3p;
@izadi2014opc].*

### 3.2 Why TIP3P is the default for method development

Three reasons, in order of weight.

1. **It is the water Sage's aqueous training data used.** Sage's Lennard-Jones parameters were
   refit against condensed-phase densities and enthalpies of mixing, and "physical properties of
   aqueous systems using TIP3P water were directly included in the training set ensuring maximal
   self-consistency between the small molecule and water interactions" [@boothroyd2023sage]. A
   protein–ligand system in TIP3P is therefore the arrangement in which the ligand's non-bonded
   parameters and the water they were fit alongside agree.
2. **It is the water the protein–ligand tooling defaults to.** The Open Free Energy OpenMM relative
   free energy protocol ships `['amber/ff14SB.xml', 'amber/tip3p_standard.xml', ...]` with an
   OpenFF small-molecule force field [@openfe_rfe_defaults]. That combination is what the recent
   large-scale industrial RBFE assessment exercised across >1700 ligands from 15 companies
   [@baumann2025openfe], and what the OpenFF 2.2.1 RBFE benchmark dataset used
   [@alibay2025openff221].
3. **It is cheaper.** TIP3P is three-site; OPC is four-site. On the ALA test system in this
   repository the same box under identical settings is 1790 particles with TIP3P and 2418 with OPC
   — a 35% increase in particle count for the same chemistry. For method development, where the
   quantity being compared is a protocol rather than a number, that cost buys nothing.

**What this does not say.** It does not say TIP3P is a good model of water. It is not: it
underestimates the dielectric constant and overestimates self-diffusion, and OPC reproduces bulk
water properties substantially better [@izadi2014opc]. The claim is narrower and it is the one the
default rests on — ff14SB and Sage's aqueous training were both done in TIP3P, so TIP3P is the
water in which the pieces of this default were assembled.

### 3.3 What is NOT claimed about Sage's benchmark

The Sage 2.0.0 protein–ligand relative binding free energy benchmark **did not use ff14SB or
ff19SB**. It used `ff99SB*-ILDN`:

> "For the ligand molecules, the Sage 2.0.0 force field was used. The protein was parametrized with
> the AMBER `ff99sb*-ILDN` force field, and a TIP3P explicit water model was employed."
> [@boothroyd2023sage]

The authors' stated reason was compatibility, not optimality: "We chose AMBER `ff99sb*-ILDN` as the
protein force field because Parsley and Sage are essentially AMBER-family force fields and should
be compatible, or nearly so, with AMBER protein force fields."

Two further corrections that this document exists partly to make:

* **Sage was not fitted to protein-binding affinities.** Binding free energies appear in that paper
  as a *benchmark*, run "to ensure the refit did not adversely affect performance on this critical
  measure" [@boothroyd2023sage]. The valence parameters were fit to quantum-chemical data from
  QCArchive; the Lennard-Jones parameters were fit to condensed-phase densities and enthalpies of
  mixing. No experimental binding affinity entered either fit.
* **Sage's complete parameter set was not fitted against TIP3P.** TIP3P entered the fit only
  through the aqueous subset of the physical-property training data, and the water model itself was
  not refit — the paper notes that the residual systematic overprediction of aqueous mixing
  enthalpies could not be removed for exactly that reason [@boothroyd2023sage]. The valence
  parameters were fit to QM and involve no water model at all.

The combination this repository defaults to — **ff14SB + Sage 2.2.1 + TIP3P** — therefore has:
**CB** evidence for *Sage 2.0 + an AMBER protein force field + TIP3P* as a class (from the Sage
paper and from [@hahn2024openff], which uses `Amber99sb*ILDN` + TIP3P with OpenFF ligand
parameters); **ID + CE** evidence for *ff14SB + TIP3P + an OpenFF small-molecule force field* as a
workflow (the OpenFE protocol default [@openfe_rfe_defaults] and the assessments run on it
[@baumann2025openfe; @alibay2025openff221]); and **EX** for the specific triple with Sage **2.2.1**,
which has no protein–ligand benchmark published under its own version number. Sage 2.2.1 differs
from 2.2.0 only in fixing "some linear angles to stay at 180 degrees", and 2.2.0 from 2.1.0 in
"small ring internal angles" and "sulfamide geometries" [@openff_forcefields_releases]; the
extrapolation from the 2.x benchmarks is small, but it is an extrapolation and is labelled one.

### 3.4 Keeping ff19SB + OPC available

`md-openmm sys-config --solvent OPC` selects ff19SB + OPC, unchanged. It is the right choice when
the science depends on the water model — conformational ensembles of intrinsically disordered or
marginally stable peptides, where ff14SB's helicity bias and TIP3P's compactness both matter, and
where ff19SB was specifically shown to improve on ff14SB [@tian2020ff19sb]. It is selected by one
flag and written into the same editable YAML; there is no profile registry and no inheritance.

**Not claimed:** that ff19SB, Sage 2.2.1 and OPC were jointly optimized. They were not. OPC was not
in Sage's training data, and no published benchmark exercises that specific triple. It is offered
as an internally consistent protein/water pair with a ligand force field bolted on, which is the
same structural compromise as the default — with the difference that the default's ligand/water
pairing has training-set support and this one does not.

---

## 4. Ligand parameters: Sage 2.2.1 and standard AM1-BCC

Sage is pinned by the resource the toolkit actually loads, `openff-2.2.1`, resolved from the
`sage-2.2.1` label a user writes. The installed `openforcefields` package ships
`openff-2.2.1.offxml`, dated 2024-09-11; `md-openmm sys-gen` records the resolved resource name in
`inputs/forcefield.json` rather than the label.

Charges are **standard AM1-BCC** through AmberTools' `sqm`, via the OpenFF toolkit registry, which
is the charge model Sage's non-bonded parameters were developed alongside [@boothroyd2023sage]. The
builder refuses to proceed if the AmberTools wrapper is absent rather than silently falling back.
`am1bcc_nagl` is available but is never a default: NAGL is a graph network *trained to predict*
AM1-BCC ELF10 charges. It is close to them and it is not that calculation, so selecting it produces
a different Hamiltonian and is recorded as such.

For a protein-only system the record states that no ligand force field participated, rather than
naming Sage for a calculation it contributed nothing to.

*Evidence: **JP** (Sage + AM1-BCC), **ID** (resource resolution).*

---

## 5. Implicit solvent: ff14SB + GBn2 + mbondi3, no surface-area term

GB-Neck2 (GBn2) was developed **with ff99SB**, and its parameters were fit to Poisson–Boltzmann
polar solvation energies and effective Born radii, not to explicit-solvent or experimental data
[@nguyen2013gbn2]. Its training and test sets were peptides, mini-proteins and proteins — Ala10,
trpzip2, HP36, tc5b, HIV-1 protease, lysozyme. It extends GB-Neck [@mongan2007gbneck] by making the
{α, β, γ} neck parameters element-dependent.

Three consequences follow directly and are implemented:

1. **The protein force field is ff14SB, not ff19SB.** ff19SB's CMAPs were fit in explicit OPC water
   and no GB model has been reparameterised against them; GBn2 belongs to the ff99SB/ff14SB
   lineage. `resolve_sys_config` **refuses** the ff19SB + GBn2 pair rather than warning about it,
   because that pair runs to completion and produces plausible numbers.
2. **No SASA term.** GBn2's parameters reproduce PB *polar* solvation; the surface-area nonpolar
   term is a separate model. Amber's `igb=8` with `gbsa=0` is the context the parameters were fit
   in, and that is what this repository builds. OpenMM's `implicit/gbn2.xml` turns the ACE term on
   by default and ParmEd leaves it off — the two differ by ~16 kJ/mol (~6 kT) on ACE-ALA-NME on
   this machine — so the choice is stated in `forcefield.json`, never inherited.
3. **The construction route is part of the Hamiltonian.** The System is built through
   `parmed.Structure.createSystem`, not `AmberPrmtopFile`, and that is recorded.

*Evidence: **JP** (GBn2 with the ff99SB/ff14SB lineage), for peptides and proteins.*

---

## 6. The limitation of GBn2/mbondi3 for arbitrary Sage ligands

This is the sharpest limitation in the repository and it is measured, not asserted.

### 6.1 mbondi3 reduces to mbondi2 for a one-residue ligand

`mbondi3` is `mbondi2` plus adjustments keyed on **residue names** — GLU, ASP, GL4, AS4 carboxylate
oxygens to 1.4 Å, ARG HH/HE hydrogens to 1.17 Å — and on the **atom name** `OXT`. A
Sage-parameterised small molecule is a single `UNL` residue with no such names, so no adjustment can
match and mbondi3 is *exactly* mbondi2 for it. `sys-gen` measures this and records
`mbondi3_reduces_to_mbondi2` in `forcefield.json`.

### 6.2 Elements outside the GB-Neck2 fit get unfitted parameters, silently

GB-Neck2's {α, β, γ} were optimized for **hydrogen, carbon, nitrogen and oxygen**; sulfur was
assigned the oxygen values after the authors "found that S parameters have insignificant effect"
[@nguyen2013gbn2]. ParmEd implements exactly that: fitted (screen, α, β, γ) for elements
{H, C, N, O, S}, and for everything else the values its own source comments call "non-optimized
values as defaults" — **α = 1.0, β = 0.8, γ = 4.85, screen = 0.5**
(`parmed.structure.Structure._get_gb_parameters`, ParmEd 4.3.1).

Fluorine, chlorine, bromine, iodine, phosphorus in a non-nucleic residue, selenium, boron and
silicon are all outside the fit. **Nothing fails.** Every atom receives a radius and a parameter
set, the System builds, the energy is finite, and no warning comes from the library. The GBn2 paper
itself flags the boundary — "extension to non-protein systems will be explored in the future"
[@nguyen2013gbn2] — but the software does not enforce it.

### 6.3 What this repository does about it

`sys-gen` reads the per-particle parameters back off the built `CustomGBForce`, counts how many
atoms carry the unfitted triple, and publishes the result:

```json
"implicit_solvent": {
  "parameter_coverage": {
    "measured": true,
    "measured_on": "openmm.CustomGBForce per-particle parameters of the built System",
    "n_atoms_with_fitted_gbn2_parameters": 18,
    "n_atoms_with_unfitted_gbn2_parameters": 1,
    "unfitted_atomic_numbers": [17],
    "mbondi3_reduces_to_mbondi2": true,
    "all_atoms_covered_by_gbn2_fit": false
  },
  "support_status": "experimental",
  "amber_igb8_parity_claimed": false
}
```

The rules, implemented in `forcefield_record._implicit_support_status`:

* **peptide/protein route, all atoms covered** → `support_status: "supported"`,
  `amber_igb8_parity_claimed: true`.
* **ligand route, all atoms covered** (a C/H/N/O/S-only drug-like molecule) →
  `support_status: "experimental"`. The elements are in the fit but the *chemistry* was not: GBn2
  was trained on peptides and proteins, and nothing published validates it for drug-like scaffolds
  with OpenFF valence and vdW parameters.
* **any atom outside the fit** → `support_status: "experimental"`, and `sys-gen` prints a warning
  naming the atomic numbers.

Exact Amber `igb=8`/`mbondi3`/`gbsa=0` parity is claimed **only** in the first case, and the record
carries the basis for the claim in `amber_igb8_parity_basis`.

*Evidence: **CE** for the peptide route; the ligand route is labelled experimental and is not
claimed at all.*

---

## 7. Box shape and the 1.5 nm padding default

### 7.1 Four distances, and they are not interchangeable

Conflating these is the most common way a box-size argument goes wrong, so the code and the record
keep them apart under names that say which is which:

1. **Requested solute-to-box clearance** (`padding_nm`). What OpenMM's `Modeller.addSolvent` takes.
   Its semantics are `width = max(2R + padding, 2·padding)` where `R` is the solute bounding radius
   — so for a compact solute the box is set by the padding alone and the solute does not enter.
2. **Solute-to-periodic-copy clearance.** How far the solute is from its own nearest image. Set by
   the **shortest lattice translation**, which equals the box width for all three shapes OpenMM
   builds. This is the quantity a self-interaction argument is about.
3. **Shortest periodic box height** — `min(a_x, b_y, c_z)` of the reduced vectors. For a rhombic
   dodecahedron this is `width/√2`, i.e. **29% smaller than the width**.
4. **Cutoff plus minimum-image margin** — `2·cutoff + margin`, the threshold quantity 3 must clear.
   OpenMM's own check refuses a cutoff larger than half the reduced-box height; this repository adds
   a 0.1 nm margin on top so the NPT contraction that immediately follows does not cross it.

`sys-gen` records all four, plus whether the box had to be grown, in
`forcefield.json → explicit_solvent.box_geometry` and in `provenance.yaml → forcefield_summary.box`.

### 7.2 Why 1.5 nm, and what it is not

1.5 nm is what production protein–ligand free-energy work uses. Hahn et al. placed their systems
"in a dodecahedral box with a minimal distance of 1.5 nm to the box wall", solvated in TIP3P at
150 mM NaCl, across the OpenFF/GAFF/CHARMM comparison [@hahn2024openff]; the OpenFF 2.2.1 RBFE
benchmark used a dodecahedron at 1.0 nm padding in the complex and 1.5 nm in solvent
[@alibay2025openff221]; the OpenFE protocol default is 1.2 nm [@openfe_rfe_defaults]. Against that,
2.0 nm — the previous default here — is generous rather than standard, and for a folded solute it
buys volume rather than physics.

The rhombic dodecahedron is chosen because it is the space-filling cell that gives the largest
minimum image distance per unit volume of the three OpenMM builds: for the same solute-to-copy
clearance it holds roughly 71% of the water a cube would.

**What 1.5 nm does not do.** It does not universally prevent self-interaction for every future
conformation. The box is fixed at build time; the solute is not. A peptide that unfolds, a
macrocycle that opens, or a REST2 ladder whose hot rungs expand the solute will all reduce the
clearance below what was requested — and PME's periodicity means the artifact is a bias in the
electrostatics, not a visible failure [@hunenberger1999ewald]. El Hage et al. showed that even a
folded protein can need a much larger box than convention suggests before its dynamics stop
depending on the box [@elhage2018boxsize].

So the guarantee this repository actually offers is narrower and enforced: **the built box is
checked against the cutoff.** `solvation._resolve_box` computes the reduced-box height of the box
that will be built, compares it to `2·cutoff + margin`, and grows the box until it clears —
recording `grown_for_cutoff: true` when it does. On the ALA test system at the default 1.5 nm the
gate is reached and passed without growth: width 3.0 nm, solute-to-copy clearance 2.053 nm, reduced
height 2.121 nm against a required 2.1 nm.

**When to use 2.0 nm** (`solvent.padding_nm: 2.0`): an unfolded or intrinsically disordered solute;
a solute expected to extend during sampling; REST2 or any enhanced sampling whose hot rungs expand
the solute; a highly charged solute where Ewald finite-size artifacts scale with the charge
[@hunenberger1999ewald]; or any case where the built box's recorded `solute_image_clearance_nm` is
uncomfortably close to the cutoff.

*Evidence: **CB** for 1.5 nm in protein–ligand production work [@hahn2024openff]; **CE + EX** for
the general case; the cutoff gate is enforced code, not evidence.*

---

## 8. PME and the 1.0 nm cutoff

Particle mesh Ewald [@darden1993pme] in its smooth form [@essmann1995spme] is the standard
treatment for periodic electrostatics and is what every reference protocol in §7 uses. A plain
cutoff on Coulomb interactions is not an option for a charged solvated system at any cutoff that is
affordable.

The 1.0 nm real-space cutoff matches the OpenFE protocol default [@openfe_rfe_defaults] and sits
inside the 1.0–1.2 nm band used across the AMBER-family literature; Hahn et al. used 1.1 nm with a
1.0–1.1 nm van der Waals switch [@hahn2024openff]. The real-space cutoff in PME is a
work-partitioning parameter for the *electrostatics* — the split between real and reciprocal space —
and its accuracy is governed by the Ewald error tolerance, here 5×10⁻⁴, which OpenMM uses to choose
the splitting parameter and grid. It is *not* only that for the **Lennard-Jones** term, which is
truncated at the same distance; the analytic long-range dispersion correction is enabled to
compensate the isotropic part of what is lost.

The full nonbonded treatment is read back off the built `NonbondedForce` and recorded — method name,
cutoff, switching function state and switch distance, dispersion correction state, Ewald error
tolerance, minimum-image margin — rather than copied from the request. The record and the System
cannot disagree, because the record is made from the System.

**Not claimed:** that 1.0 nm is converged for any particular observable. Lennard-Jones truncation at
1.0 nm with an isotropic correction is an approximation whose error is system-dependent, and the
correction assumes a homogeneous fluid beyond the cutoff, which a solvated protein is not.

*Evidence: **CE** (PME), **ID + CE** (the cutoff value).*

---

## 9. Langevin-middle integration at 300 K with 1.0 ps⁻¹ friction

OpenMM's `LangevinMiddleIntegrator` implements the LFMiddle discretization [@zhang2019lfmiddle],
which its own documentation describes as closely related to BAOAB [@leimkuhler2016geodesic] —
identical trajectories, half-step versus on-step velocities. The middle-scheme family was
constructed specifically to reduce the configurational sampling error of Langevin integrators at a
given timestep [@leimkuhler2012rational; @leimkuhler2013baoab], and that is the property this
repository wants: at 2 fs and 4 fs the configurational distribution should be as close to correct as
the discretization allows.

`friction_per_ps: 1.0` is OpenMM's collision rate in ps⁻¹, corresponding to a nominal 1 ps
damping time. It is written as `friction_per_ps` and not as a prose "collision time" precisely
because the two are reciprocal and the ambiguity is a real source of factor-of-N errors between
codes. 1.0 ps⁻¹ is the value the OpenFE protocol uses [@openfe_rfe_defaults] and sits in the weak
regime: strong enough to thermostat and to damp the energy a barostat move injects, weak enough
that the dynamics between collisions are close to Newtonian.

**The distinction that matters.** Langevin dynamics samples the correct canonical *configurational*
distribution for any positive friction. It does **not** leave real-time dynamics unchanged. The
friction coefficient enters the equations of motion; diffusion constants, velocity and dipole
autocorrelation functions, barrier-crossing rates and any transport property computed from a
Langevin trajectory depend on it, and in the low-friction and high-friction limits they depend on it
in opposite directions. **Thermostat choice is not irrelevant to kinetics.** Anyone extracting a
rate, a diffusion coefficient or a relaxation time from a run generated by these templates must
treat the friction as a parameter of the result, not as a numerical detail — and should verify the
observable's dependence on it rather than assume 1.0 ps⁻¹ is neutral.

*Evidence: **CE** (the integrator), **CE + EX** (the friction value).*

---

## 10. Monte Carlo barostat, 1 bar, 25-step attempt interval

The Monte Carlo barostat samples the isothermal–isobaric ensemble by proposing volume changes and
accepting them with a Metropolis criterion that includes the `pV` and `N ln V` terms
[@chow1995mcbarostat; @aqvist2004mcbarostat]. It has no fictitious barostat degree of freedom and no
time constant to tune, so unlike an extended-Lagrangian barostat it cannot introduce spurious
oscillations in the volume — which is why it is the appropriate choice for a template whose users
will not tune it.

**`barostat_frequency_steps: 25`** is now a public setting. 25 steps is OpenMM's own default —
verified directly on the installed build, `MonteCarloBarostat(1 bar, 300 K).getFrequency() == 25` —
and is what the OpenFE protocol uses [@openfe_rfe_defaults]. It is an interval in **integration
steps**, not in time, so its physical meaning depends on the timestep:

| timestep | 25 steps is |
|---|---|
| 2 fs (default) | 0.05 ps |
| 4 fs (HMR option) | 0.10 ps |

Both numbers are recorded: every `resolved_stage.yaml` carries `barostat_frequency_steps` and
`barostat_interval_ps`, and `MD/provenance.yaml` carries both plus the list of stages in which the
barostat is active.

**Not claimed:** that 25 steps is optimal. It is frequent enough that the volume equilibrates
quickly and infrequent enough that the cost of the extra energy evaluations is small; it is not
derived from any convergence study, and the correlation time of the volume in a given system is not
something a default can know.

### The invariants the implementation preserves

* **minimization** — barostat present in the System, frequency 0. A volume move against forces that
  are still enormous is not a physical proposal.
* **NVT** — barostat present, frequency 0.
* **explicit NPT** — exactly one active `MonteCarloBarostat`.
* **implicit solvent** — no barostat in the System at all, `barostat_frequency_steps: null`,
  ensembles reconciled to NVT, `pressure_bar: null`.
* **each REST2 replica** — its own barostat with a distinct deterministic seed, at the one
  configured frequency.

The barostat is present-but-inert in the non-NPT explicit stages deliberately: the Force layout, and
therefore the checkpoint layout, stays identical along the whole chain. **Presence is not activity**,
and the runtime records report both `barostats_in_system` and `barostats_active` so a reader never
has to infer one from the other.

For REST2 specifically: every replica shares one thermostat temperature and one β and differs only
by Hamiltonian, so the `pV` contributions cancel in the NPT exchange criterion, and positions and
box vectors travel together as one configuration [@wang2011rest2].

*Evidence: **CE** (the algorithm), **ID** (the value 25).*

---

## 11. 2 fs as the baseline; HMR + 4 fs as an explicit option

### 11.1 Why 2 fs is the default

With `constraints=HBonds` the bond stretches involving hydrogen are removed, and the fastest
remaining motions are the bond-angle vibrations involving hydrogen, with periods around 10 fs. A 2 fs
step resolves those with a comfortable margin and is the long-standing default across the
AMBER-family literature, including every validation run in the ff14SB paper [@maier2015ff14sb].

Crucially, at 2 fs **the masses are the ones the force field assigned**. Nothing about the dynamics
is altered to buy the timestep. For a repository whose purpose is method development — comparing
protocols, not maximising throughput — that is the right trade, and it means a trajectory produced
by the defaults can be used for kinetic analysis without a caveat about its inertia.

### 11.2 The HMR option

`docs/examples/hmr-4fs.yaml` gives the complete change: hydrogen mass 3.024 amu, timestep 4 fs,
`constraints: HBonds`, `rigid_water: true`. Repartitioning moves mass from each heavy atom onto the
hydrogens bonded to it, lowering the hydrogen angle-bend frequencies and permitting a longer step
[@feenstra1999hmr]. Hopkins et al. showed that with a hydrogen mass around 3 amu, 4 fs integration is
stable for biomolecular systems and reproduces equilibrium structural and thermodynamic properties
[@hopkins2015hmr]. The OpenFE protocol runs 4 fs with 3.0 amu hydrogens as its production default
[@openfe_rfe_defaults], and the industrial RBFE assessment [@baumann2025openfe] was run on that
protocol — so the setting has genuine large-scale support **for equilibrium free-energy work**.

### 11.3 The four things that must be kept apart

1. **Stability at 4 fs.** Well supported [@feenstra1999hmr; @hopkins2015hmr] and, for suitable
   biomolecular systems, effectively standard practice.
2. **Equilibrium configurational sampling.** The Boltzmann distribution in configuration space does
   not depend on masses at all, so equilibrium structural and thermodynamic averages — free
   energies included — are unaffected by the repartitioning itself. Discretization error from the
   longer step is a separate question and is not zero.
3. **Altered masses.** The inertia of every hydrogen-bearing group is changed. This is not a
   numerical trick applied to the integrator; it is a change to the system being integrated.
4. **Kinetics, time correlation functions, diffusion and transport.** These are computed from the
   dynamics, and the dynamics depend on the masses. Rates, relaxation times, diffusion constants and
   spectral densities from an HMR trajectory are **not** the same quantities as from an
   unrepartitioned one, and no amount of stability testing makes them so.

**A short stable smoke run is not validation.** This repository's own CUDA acceptance run for the
HMR path is picoseconds long; it establishes that the constraint/mass combination integrates without
diverging and that the masses are what the record says. It establishes nothing about kinetics.
Before using 4 fs on a system class you have not run before, compare a few hundred picoseconds
against the 2 fs baseline — total energy drift, temperature, density, and whichever structural
quantity your project cares about.

### 11.4 What the implementation enforces

* 4 fs with `hydrogen_mass_amu: null` is **refused**, naming both files.
* 4 fs with hydrogens constrained but `constraints.type` set to `None` is **refused**: HMR lowers
  the angle frequencies and does not touch the bond stretches, which would then be the fastest
  motions left.
* 4 fs with `rigid_water: false` under explicit solvent is **refused**: water is never
  repartitioned, so a flexible water keeps its ~10 fs O–H stretch and sets the stable timestep for
  the whole box on its own.
* A target mass below 1.008 amu is refused; a target that would drive a heavy atom to a nonpositive
  mass is refused.
* **Mass conservation is verified group by group**, against the same `createSystem` call with
  `hydrogenMass` omitted. For each heavy atom and the hydrogens bonded to it, the total before and
  after must agree; checking only the box total would let two errors of opposite sign cancel. The
  result is published as `constraints.hmr_group_conservation` in `forcefield.json`.

*Evidence: **CB** for stability, **CE** for the equilibrium-sampling argument, and explicitly
**nothing** for kinetic equivalence.*

---

## 12. What these citations do not establish

Read this section before quoting anything above as support for a result.

1. **No citation validates the specific triple ff14SB + Sage 2.2.1 + TIP3P.** The Sage
   protein–ligand benchmark used `ff99SB*-ILDN` [@boothroyd2023sage]. The ff14SB + TIP3P + OpenFF
   workflow evidence comes from software defaults [@openfe_rfe_defaults] and assessments run on
   them [@baumann2025openfe; @alibay2025openff221], one of which is a preprint and one of which is a
   dataset. The step from "Sage 2.x with an AMBER protein force field in TIP3P" to "Sage 2.2.1 with
   ff14SB in TIP3P" is an extrapolation.
2. **No citation validates ff19SB + Sage 2.2.1 + OPC.** ff19SB + OPC is jointly recommended
   [@tian2020ff19sb]; Sage with OPC is not benchmarked anywhere, and OPC was not in Sage's training
   data. The alternative is offered as two consistent halves, not as a validated whole.
3. **TIP3P is not endorsed as a water model.** §3.2 argues only that it is the water in which
   ff14SB was corrected and Sage's aqueous properties were fit. Its bulk properties are worse than
   OPC's [@izadi2014opc], and ff14SB's agreement with experiment depends partly on cancellation with
   that fact [@tian2020ff19sb].
4. **GBn2 is not validated for non-peptidic chemistry.** The GBn2 paper says so
   [@nguyen2013gbn2], and §6 measures the consequence on the built System. The implicit ligand route
   is labelled experimental in the record and in this document, and no `igb=8` parity is claimed for
   it.
5. **1.5 nm padding is not a guarantee.** It is common practice in protein–ligand production work
   [@hahn2024openff] for solutes that stay folded. Nothing about it bounds the self-interaction of a
   conformation that has not happened yet [@elhage2018boxsize].
6. **1.0 nm is not a converged cutoff for any named observable.** It matches reference protocols.
   No convergence study was run here.
7. **1.0 ps⁻¹ friction is not neutral for kinetics.** §9.
8. **25 steps is not an optimised barostat frequency.** It is OpenMM's default.
9. **HMR + 4 fs is not validated for quantitative kinetics.** §11.3. It is supported for stability
   and for equilibrium free energies.
10. **No number in this repository has been validated against experiment.** Every benchmark cited
    was run by someone else, on their systems, with their protocol. The tests in this repository
    check that the implemented values are the documented values and that the built Systems carry
    them; that is a different claim, and it is the only one this repository makes.
11. **The Sage 2.0 benchmark and the OpenFE assessments are relative binding free energy studies.**
    They say something about alchemical free-energy accuracy for congeneric series. They say nothing
    directly about conventional MD or REST2 conformational sampling, which is what these templates
    generate.
12. **Two ff14SB conversions exist and this repository uses one of them.** OpenMM ships
    `amber14/protein.ff14SB.xml` (included by `amber14-all.xml`, which is what this repository
    loads); openmmforcefields ships `amber/ff14SB.xml`, which is what OpenFE loads. Both are ff14SB.
    They have not been compared parameter-by-parameter here, and no claim of byte or numerical
    identity is made.

---

## 13. Recommendation for method-development comparisons

If you are comparing protocols — sampling methods, restraint schemes, REST2 ladders, timestep
choices — rather than predicting a number:

1. **Use the defaults, unchanged, for every arm of the comparison.** ff14SB + Sage 2.2.1 + TIP3P,
   dodecahedron, 1.5 nm, PME at 1.0 nm, Langevin-middle at 1.0 ps⁻¹, 2 fs, no HMR, barostat every
   25 steps. Their virtue here is not that they are the most accurate available; it is that they are
   internally consistent, cheap, and identical across arms, so a difference between arms is a
   difference in the method.
2. **Do not enable HMR for an arm whose observable is dynamical.** If any arm reports a rate, a
   relaxation time or a diffusion constant, keep every arm at 2 fs with unmodified masses.
3. **Raise padding to 2.0 nm for every arm if any arm expands the solute.** A REST2 ladder and a
   cMD walker compared in differently sized boxes are not the same comparison. Set it once, in
   `sys.config.yaml`, before `sys-gen`.
4. **Switch to ff19SB + OPC when the water model is the variable**, or when the science is a
   conformational ensemble of a disordered or marginally stable peptide — and switch every arm.
5. **Archive `inputs/` with the trajectories.** `system.xml` records what was built; the
   configuration records only what was requested. An archive holding trajectories alone cannot say
   which Hamiltonian produced them.

---

## 14. Reproducibility: where each of these values is recorded

Every choice argued above is written into the generated records, at the point it is decided, and
never reconstructed by parsing a log afterwards. `null` means "does not apply here" and is written
explicitly; `unknown` means "not recorded" and is never upgraded to a guess.

| this document | the record |
|---|---|
| protein force field, and which file inside a wrapper carried it | `inputs/forcefield.json → protein.openmm_resource`, `protein.openmm_resource_includes` |
| water model and the qualified resource loaded | `→ water.model`, `water.openmm_resource`; packing substitution under `explicit_solvent.water_packing_*` |
| ligand force field and charge route | `→ ligand.openff_resource`, `ligand.charge_method`, `ligand.charge_model` |
| implicit model, radii, SASA state, GB coverage, support status | `→ implicit_solvent.*`, including `parameter_coverage` and `amber_igb8_parity_claimed` |
| the full nonbonded treatment (§8) | `→ nonbonded.{method, cutoff_nm, switching, switch_distance_nm, dispersion_correction, ewald_error_tolerance}` |
| the four box distances (§7.1) | `→ explicit_solvent.box_geometry.*`, plus `box_vectors_nm` and `box_volume_nm3` |
| salt versus neutralising counterions | `→ explicit_solvent.salt` |
| constraints, hydrogen masses, HMR group conservation | `→ constraints.*`, `constraints.hmr_group_conservation` |
| thermostat, friction, timestep | `MD/provenance.yaml → protocol.thermostat`, `protocol.timestep_fs`; each `stage.yaml`; each `resolved_stage.yaml` |
| barostat pressure, frequency in steps and in ps, which stages it is active in | `MD/provenance.yaml → protocol.pressure_coupling`; `resolved_stage.yaml → barostat_frequency_steps`, `barostat_interval_ps`, `barostats_in_system`, `barostats_active`; `resolved_run.yaml → pressure_coupling` |
| every package version that could change a parameter | `forcefield.json → package_versions`; `provenance.yaml → environment` |
| what was built, as one summary beside the lineage hashes | `inputs/provenance.yaml → forcefield_summary` |

`sys-gen` additionally keeps the user's original input byte-for-byte under `original_inputs/`, keeps
the construction artifacts that carry science under `preparation/`, and writes `SHA256SUMS` over the
whole directory. `md-gen` records the parent system's hashes, the stage plan, every seed and
`generated-files.sha256`.

**0.3.x data is not retrofitted.** The defaults changed in 0.4; what 0.3.x ran did not. The
retrospective tool `scripts/retrofit_fair_v030.py` reads only, preserves a recorded ff19SB/OPC
identity exactly, and leaves a missing value as `unknown` rather than filling it from the current
default. Two tests assert this directly.

---

## 15. Sources

Full entries, with the DOI of each verified against the Crossref REST API on 2026-08-27, are in
`docs/md-defaults-references.bib`. Software documentation and dataset entries are labelled as such
there and in §2.

Primary literature: [@maier2015ff14sb] ff14SB · [@tian2020ff19sb] ff19SB · [@jorgensen1983tip3p]
TIP3P · [@izadi2014opc] OPC · [@boothroyd2023sage] Sage 2.0.0 · [@nguyen2013gbn2] GB-Neck2 ·
[@mongan2007gbneck] GB-Neck · [@darden1993pme; @essmann1995spme] PME · [@hunenberger1999ewald]
Ewald artifacts · [@elhage2018boxsize] box size · [@leimkuhler2012rational; @leimkuhler2013baoab;
@leimkuhler2016geodesic] BAOAB/geodesic Langevin · [@zhang2019lfmiddle] LFMiddle ·
[@chow1995mcbarostat; @aqvist2004mcbarostat] Monte Carlo pressure coupling · [@feenstra1999hmr;
@hopkins2015hmr] HMR · [@wang2011rest2] REST2 · [@hahn2024openff] open-source force fields in
protein–ligand affinity prediction · [@eastman2017openmm7; @eastman2024openmm8] OpenMM ·
[@parmed] ParmEd.

Software documentation and datasets, used only for implementation facts:
[@openfe_rfe_defaults; @openmm_userguide; @openff_forcefields_releases; @alibay2025openff221].
Preprint, not peer reviewed: [@baumann2025openfe].
