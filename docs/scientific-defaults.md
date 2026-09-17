# Scientific rationale for the MD-tools OpenMM defaults

| | |
|---|---|
| Repository | MD-tools (`csy0000/MD-tools`), version 0.5.4 |
| Applies to | the defaults `md-openmm build-top` and `md-openmm build-md` apply |
| Date | 2026-08-27 |
| References | `docs/scientific-defaults.bib` |

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
| **DNI** — documented, not implemented | an option discussed here for completeness that this repository does **not** provide |

An **ID** source is never used in place of a primary source for a scientific claim. Where a value
is taken from software documentation because it *is* a software fact — OpenMM's barostat default
frequency, for instance — that is said plainly.

---

## 2. The defaults, in one table

| setting | default | alternative | evidence | limitation |
|---|---|---|---|---|
| protein force field | ff14SB (`amber14-all.xml` → `amber14/protein.ff14SB.xml`) | ff19SB (`amber19-all.xml`) | **JP** with TIP3P (§3) | backbone correction is empirical and TIP3P-specific |
| water | TIP3P (`amber14/tip3p.xml`) | OPC (`amber19/opc.xml`) | **JP** (§3) | TIP3P misrepresents bulk water; ff14SB relies on error cancellation with it |
| ligand force field | OpenFF Sage 2.2.1 (`openff-2.2.1`) | GAFF, resolved to an exact version (§4.1) | **CE** + **CB** at Sage 2.0 (§4) | 2.2.1 itself has no published protein–ligand benchmark; GAFF2 has no dedicated publication |
| ligand charges | AM1-BCC via AmberTools `sqm` | `am1bcc_nagl` (explicit) | **JP** with Sage (§4) | NAGL predicts AM1-BCC ELF10, it does not compute it |
| implicit solvent | GBn2 + mbondi3, no SASA | — | **JP** with ff99SB/ff14SB (§5) | peptides and proteins only; see §6 |
| box | rhombic dodecahedron | `cube`, `octahedron` | **CE** (§7) | a cube needs ~1.4× the water for the same clearance |
| padding | 1.5 nm | 2.0 nm | **CB** (§7) | not a guarantee for any future conformation |
| electrostatics | PME, 1.0 nm real-space cutoff | — | **CE** + **ID** (§8) | finite-size artifacts are reduced, not removed |
| thermostat | `LangevinMiddleIntegrator`, 300 K | — | **CE** (§9) | — |
| friction | 1.0 ps⁻¹ | — | **CE** + **EX** (§9) | affects kinetics and transport, not just sampling speed |
| barostat | `MonteCarloBarostat`, 1 bar | Berendsen: **documented, not implemented** (§10.1) | **CE** (§10) | Berendsen damps volume fluctuations and is not offered |
| barostat frequency | 25 steps | — | **ID** + **CE** (§10) | 25 is OpenMM's default, not an optimum for any system |
| timestep | `auto` → 2 fs, or 4 fs on a repartitioned System (§11.5) | any explicit value ≤ 3 fs | **CE** (§11) | resolved from the System's masses, never from a configuration's claim |
| hydrogen mass | unmodified (`enabled: false`) | 3.024 amu (`enabled: true`) | **CB** for stability (§11) | dynamics, kinetics and transport are altered |
| ionic strength | 0.15 M NaCl, counterions separate | — | **CE** | ion parameters are whatever the water XML carries |

---

## 3. ff14SB + TIP3P as the explicit default, ff19SB + OPC as the alternative

### 3.1 The two pairs are each internally consistent, and they are not interchangeable

ff14SB's side-chain dihedrals were fit to gas-phase QM (`HF/6-31G*` geometries, `MP2/6-31+G**`
single points), but its backbone adjustment was not: it is, in the authors' words, "an empirical
correction based on data obtained from simulations carried out in TIP3P explicit water"
[1]. Every validation simulation in that paper used TIP3P, and the paper closes the
point with an explicit caution — "transferability of this correction term to other implicit or
explicit solvent models may need evaluation prior to production use."

ff19SB replaced that empirical backbone correction with amino-acid-specific CMAP maps trained
against QM in aqueous (implicit) solvent, and its authors recommend a different water model:
"Of the explicit water models tested here, we recommend use of OPC with ff19SB" [2].
The same paper is blunt about why the older pair works: ff14SB has an inherent underestimation of
helicity that TIP3P's compact-structure bias partly cancels.

So the two pairs are not two spellings of the same thing. **ff14SB + TIP3P** is a pair whose known
error cancellation is the reason it reproduces the benchmarks it reproduces; **ff19SB + OPC** is a
pair whose backbone was fit to be right for a better water model. Mixing them across — ff19SB with
TIP3P, or ff14SB with OPC — discards the reason either works. This repository therefore ships the
protein force field and the water model as **one selection**, not as two independent keys with
independent defaults.

*Evidence: **JP** for each pair. [1, 2, 3, 4].*

### 3.2 Why TIP3P is the default for method development

Three reasons, in order of weight.

1. **It is the water Sage's aqueous training data used.** Sage's Lennard-Jones parameters were
   refit against condensed-phase densities and enthalpies of mixing, and "physical properties of
   aqueous systems using TIP3P water were directly included in the training set ensuring maximal
   self-consistency between the small molecule and water interactions" [5]. A
   protein–ligand system in TIP3P is therefore the arrangement in which the ligand's non-bonded
   parameters and the water they were fit alongside agree.
2. **It is the water the protein–ligand tooling defaults to.** The Open Free Energy OpenMM relative
   free energy protocol ships `['amber/ff14SB.xml', 'amber/tip3p_standard.xml', ...]` with an
   OpenFF small-molecule force field [6]. That combination is what the recent
   large-scale industrial RBFE assessment exercised across >1700 ligands from 15 companies
   [7], and what the OpenFF 2.2.1 RBFE benchmark dataset used
   [8].
3. **It is cheaper.** TIP3P is three-site; OPC is four-site. On the ALA test system in this
   repository the same box under identical settings is 1790 particles with TIP3P and 2418 with OPC
   — a 35% increase in particle count for the same chemistry. For method development, where the
   quantity being compared is a protocol rather than a number, that cost buys nothing.

**What this does not say.** It does not say TIP3P is a good model of water. It is not: it
underestimates the dielectric constant and overestimates self-diffusion, and OPC reproduces bulk
water properties substantially better [4]. The claim is narrower and it is the one the
default rests on — ff14SB and Sage's aqueous training were both done in TIP3P, so TIP3P is the
water in which the pieces of this default were assembled.

### 3.3 What is NOT claimed about Sage's benchmark

The Sage 2.0.0 protein–ligand relative binding free energy benchmark **did not use ff14SB or
ff19SB**. It used `ff99SB*-ILDN`:

> "For the ligand molecules, the Sage 2.0.0 force field was used. The protein was parametrized with
> the AMBER `ff99sb*-ILDN` force field, and a TIP3P explicit water model was employed."
> [5]

The authors' stated reason was compatibility, not optimality: "We chose AMBER `ff99sb*-ILDN` as the
protein force field because Parsley and Sage are essentially AMBER-family force fields and should
be compatible, or nearly so, with AMBER protein force fields."

Two further corrections that this document exists partly to make:

* **Sage was not fitted to protein-binding affinities.** Binding free energies appear in that paper
  as a *benchmark*, run "to ensure the refit did not adversely affect performance on this critical
  measure" [5]. The valence parameters were fit to quantum-chemical data from
  QCArchive; the Lennard-Jones parameters were fit to condensed-phase densities and enthalpies of
  mixing. No experimental binding affinity entered either fit.
* **Sage's complete parameter set was not fitted against TIP3P.** TIP3P entered the fit only
  through the aqueous subset of the physical-property training data, and the water model itself was
  not refit — the paper notes that the residual systematic overprediction of aqueous mixing
  enthalpies could not be removed for exactly that reason [5]. The valence
  parameters were fit to QM and involve no water model at all.

The combination this repository defaults to — **ff14SB + Sage 2.2.1 + TIP3P** — therefore has:
**CB** evidence for *Sage 2.0 + an AMBER protein force field + TIP3P* as a class (from the Sage
paper and from [9], which uses `Amber99sb*ILDN` + TIP3P with OpenFF ligand
parameters); **ID + CE** evidence for *ff14SB + TIP3P + an OpenFF small-molecule force field* as a
workflow (the OpenFE protocol default [6] and the assessments run on it
[7, 8]); and **EX** for the specific triple with Sage **2.2.1**,
which has no protein–ligand benchmark published under its own version number. Sage 2.2.1 differs
from 2.2.0 only in fixing "some linear angles to stay at 180 degrees", and 2.2.0 from 2.1.0 in
"small ring internal angles" and "sulfamide geometries" [10]; the
extrapolation from the 2.x benchmarks is small, but it is an extrapolation and is labelled one.

### 3.4 Keeping ff19SB + OPC available

Setting `forcefield.protein: ff19SB` and `solvent.model: OPC` selects ff19SB + OPC, unchanged. It is the right choice when
the science depends on the water model — conformational ensembles of intrinsically disordered or
marginally stable peptides, where ff14SB's helicity bias and TIP3P's compactness both matter, and
where ff19SB was specifically shown to improve on ff14SB [2]. It is selected by one
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
`openff-2.2.1.offxml`, dated 2024-09-11; `md-openmm build-top` records the resolved resource name in
`inputs/forcefield.json` rather than the label.

Charges are **standard AM1-BCC** through AmberTools' `sqm`, via the OpenFF toolkit registry, which
is the charge model Sage's non-bonded parameters were developed alongside [5]. The
builder refuses to proceed if the AmberTools wrapper is absent rather than silently falling back.
`am1bcc_nagl` is available but is never a default: NAGL is a graph network *trained to predict*
AM1-BCC ELF10 charges. It is close to them and it is not that calculation, so selecting it produces
a different Hamiltonian and is recorded as such.

For a protein-only system the record states that no ligand force field participated, rather than
naming Sage for a calculation it contributed nothing to.

*Evidence: **JP** (Sage + AM1-BCC), **ID** (resource resolution).*

---

### 4.1 GAFF as the documented alternative, resolved to an exact version

Sage remains the default. GAFF is available through `openmmforcefields`'
`GAFFTemplateGenerator`, which types the molecule with antechamber [11] and
takes the AM1-BCC charges already assigned to it [12]. GAFF version 1 is
published [13].

**GAFF2 is not.** It has been distributed with AmberTools since 2015 and its provenance is the
parameter file, not a paper [14]. Citing Wang 2004 for a GAFF2 run would
attribute version 2 parameters to the version 1 publication, which is why the two entries in the
bibliography are kept apart.

That has a direct consequence for what gets recorded. "GAFF2" does not identify a Hamiltonian:
this environment ships `gaff-2.1`, `gaff-2.11` and `gaff-2.2.20`, they differ, and a trajectory
belongs to exactly one of them. So `solute.ligand_forcefield: gaff2` is accepted as an alias and
**resolved immediately** to the newest installed 2.x; the exact version is what reaches
`built.log`, the force-field record and the machine record, and the alias never does. An exact
version may be written instead, and one that is not installed is refused with the list that is.

Because GAFF's parameters arrive through antechamber rather than from a single self-contained
file, the AmberTools build is part of the provenance in a way it is not for SMIRNOFF. The record
therefore carries the antechamber and `sqm` paths and the AmberTools version, read from the conda
package record rather than scraped from a program banner.

**Not claimed:** that GAFF2 and Sage are interchangeable, or that either is better here. They are
different Hamiltonians; a study that switches between them is comparing force fields, not
continuing a series.

*Evidence: **CE** for GAFF v1; **ID** for the GAFF2 distribution and the resolved version.*

---

## 5. Implicit solvent: ff14SB + GBn2 + mbondi3, no surface-area term

GB-Neck2 (GBn2) was developed **with ff99SB**, and its parameters were fit to Poisson–Boltzmann
polar solvation energies and effective Born radii, not to explicit-solvent or experimental data
[15]. Its training and test sets were peptides, mini-proteins and proteins — Ala10,
trpzip2, HP36, tc5b, HIV-1 protease, lysozyme. It extends GB-Neck [16] by making the
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
match and mbondi3 is *exactly* mbondi2 for it. `build-top` measures this and records
`mbondi3_reduces_to_mbondi2` in `forcefield.json`.

### 6.2 Elements outside the GB-Neck2 fit get unfitted parameters, silently

GB-Neck2's {α, β, γ} were optimized for **hydrogen, carbon, nitrogen and oxygen**; sulfur was
assigned the oxygen values after the authors "found that S parameters have insignificant effect"
[15]. ParmEd implements exactly that: fitted (screen, α, β, γ) for elements
{H, C, N, O, S}, and for everything else the values its own source comments call "non-optimized
values as defaults" — **α = 1.0, β = 0.8, γ = 4.85, screen = 0.5**
(`parmed.structure.Structure._get_gb_parameters`, ParmEd 4.3.1).

Fluorine, chlorine, bromine, iodine, phosphorus in a non-nucleic residue, selenium, boron and
silicon are all outside the fit. **Nothing fails.** Every atom receives a radius and a parameter
set, the System builds, the energy is finite, and no warning comes from the library. The GBn2 paper
itself flags the boundary — "extension to non-protein systems will be explored in the future"
[15] — but the software does not enforce it.

### 6.3 What this repository does about it

`build-top` reads the per-particle parameters back off the built `CustomGBForce`, counts how many
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
* **any atom outside the fit** → `support_status: "experimental"`, and `build-top` prints a warning
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

`build-top` records all four, plus whether the box had to be grown, in
`forcefield.json → explicit_solvent.box_geometry` and in `provenance.yaml → forcefield_summary.box`.

### 7.2 Why 1.5 nm, and what it is not

1.5 nm is what production protein–ligand free-energy work uses. Hahn et al. placed their systems
"in a dodecahedral box with a minimal distance of 1.5 nm to the box wall", solvated in TIP3P at
150 mM NaCl, across the OpenFF/GAFF/CHARMM comparison [9]; the OpenFF 2.2.1 RBFE
benchmark used a dodecahedron at 1.0 nm padding in the complex and 1.5 nm in solvent
[8]; the OpenFE protocol default is 1.2 nm [6]. Against that,
2.0 nm — the previous default here — is generous rather than standard, and for a folded solute it
buys volume rather than physics.

The rhombic dodecahedron is chosen because it is the space-filling cell that gives the largest
minimum image distance per unit volume of the three OpenMM builds: for the same solute-to-copy
clearance it holds roughly 71% of the water a cube would.

**What 1.5 nm does not do.** It does not universally prevent self-interaction for every future
conformation. The box is fixed at build time; the solute is not. A peptide that unfolds, a
macrocycle that opens, or a REST2 ladder whose hot rungs expand the solute will all reduce the
clearance below what was requested — and PME's periodicity means the artifact is a bias in the
electrostatics, not a visible failure [17]. El Hage et al. showed that even a
folded protein can need a much larger box than convention suggests before its dynamics stop
depending on the box [18].

So the guarantee this repository actually offers is narrower and enforced: **the built box is
checked against the cutoff.** `solvation._resolve_box` computes the reduced-box height of the box
that will be built, compares it to `2·cutoff + margin`, and grows the box until it clears —
recording `grown_for_cutoff: true` when it does. On the ALA test system at the default 1.5 nm the
gate is reached and passed without growth: width 3.0 nm, solute-to-copy clearance 2.053 nm, reduced
height 2.121 nm against a required 2.1 nm.

**When to use 2.0 nm** (`solvent.padding_nm: 2.0`): an unfolded or intrinsically disordered solute;
a solute expected to extend during sampling; REST2 or any enhanced sampling whose hot rungs expand
the solute; a highly charged solute where Ewald finite-size artifacts scale with the charge
[17]; or any case where the built box's recorded `solute_image_clearance_nm` is
uncomfortably close to the cutoff.

*Evidence: **CB** for 1.5 nm in protein–ligand production work [9]; **CE + EX** for
the general case; the cutoff gate is enforced code, not evidence.*

---

### 7.3 The two documented alternatives

`solvent.box_shape` accepts three values, and an unknown one is refused by the schema before
anything is solvated.

| shape | height fraction | water for the same clearance | when |
|---|---|---|---|
| `dodecahedron` (default) | `1/√2` ≈ 0.707 | ~71 % of a cube | almost always |
| `cube` | 1 | 100 % | when a downstream tool assumes an axis-aligned box |
| `octahedron` | `√6/3` ≈ 0.816 | ~82 % of a cube | familiar from Amber workflows |

The dodecahedron is the default because it is the cheapest of the three for a given
solute-to-image clearance — the quantity that actually matters — and OpenMM supports all three
directly, so this is a choice among supported reduced forms rather than a bespoke construction.

Both the requested and the realised geometry are recorded in `built.log`: the box is only ever
*grown* to satisfy the cutoff, never shrunk, and `grown_for_cutoff` says whether that happened.
A reader therefore never has to infer whether the padding they asked for is the padding they got.

*Evidence: **ID** for the supported shapes, **CE** for the clearance arithmetic (§7.1).*

---

## 8. PME and the 1.0 nm cutoff

Particle mesh Ewald [19] in its smooth form [20] is the standard
treatment for periodic electrostatics and is what every reference protocol in §7 uses. A plain
cutoff on Coulomb interactions is not an option for a charged solvated system at any cutoff that is
affordable.

The 1.0 nm real-space cutoff matches the OpenFE protocol default [6] and sits
inside the 1.0–1.2 nm band used across the AMBER-family literature; Hahn et al. used 1.1 nm with a
1.0–1.1 nm van der Waals switch [9]. The real-space cutoff in PME is a
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

OpenMM's `LangevinMiddleIntegrator` implements the LFMiddle discretization [21],
which its own documentation describes as closely related to BAOAB [22] —
identical trajectories, half-step versus on-step velocities. The middle-scheme family was
constructed specifically to reduce the configurational sampling error of Langevin integrators at a
given timestep [23, 24], and that is the property this
repository wants: at 2 fs and 4 fs the configurational distribution should be as close to correct as
the discretization allows.

`friction_per_ps: 1.0` is OpenMM's collision rate in ps⁻¹, corresponding to a nominal 1 ps
damping time. It is written as `friction_per_ps` and not as a prose "collision time" precisely
because the two are reciprocal and the ambiguity is a real source of factor-of-N errors between
codes. 1.0 ps⁻¹ is the value the OpenFE protocol uses [6] and sits in the weak
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
[25, 26]. It has no fictitious barostat degree of freedom and no
time constant to tune, so unlike an extended-Lagrangian barostat it cannot introduce spurious
oscillations in the volume — which is why it is the appropriate choice for a template whose users
will not tune it.

**`barostat_frequency_steps: 25`** is now a public setting. 25 steps is OpenMM's own default —
verified directly on the installed build, `MonteCarloBarostat(1 bar, 300 K).getFrequency() == 25` —
and is what the OpenFE protocol uses [6]. It is an interval in **integration
steps**, not in time, so its physical meaning depends on the timestep:

| timestep | 25 steps is |
|---|---|
| 2 fs (default) | 0.05 ps |
| 4 fs (HMR option) | 0.10 ps |

The requested value is recorded in `resolved.config` as `dynamics.barostat_interval_steps`, and
each stage's machine record carries what the built System actually received. (This paragraph named
`resolved_stage.yaml` and `MD/provenance.yaml`, neither of which is written by anything — see
`docs/backlog.md` entry 10.)

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
box vectors travel together as one configuration [27].

*Evidence: **CE** (the algorithm), **ID** (the value 25).*

---

### 10.1 Berendsen pressure coupling: documented, not implemented

Berendsen weak coupling rescales the box toward the target pressure with a first-order relaxation
of time constant τ_p, so the volume approaches its target smoothly and quickly
[28]. That is genuinely useful, and it is why the method survives in
equilibration protocols: relaxing a box that was built by a solvation heuristic is exactly the job
it does well.

It is not a production barostat. First-order relaxation damps the volume fluctuations rather than
generating them, so the trajectory does not sample the isothermal–isobaric distribution: the mean
volume can be right while the fluctuations around it — and therefore every quantity derived from
them, the isothermal compressibility first among them — are wrong. This is the standard position in
the literature, stated plainly in the paper that introduced stochastic cell rescaling to fix it:
the usual recipe of a Berendsen barostat for equilibration followed by a second-order or Monte
Carlo barostat for production exists precisely because the first "results in incorrect volume
fluctuations" [29].

**MD-tools does not provide it.** OpenMM ships no `BerendsenBarostat` class, and this repository
will not offer a hand-written one under that name. A pressure-control method that silently
misreports fluctuations is worse than an absent one, because a run that uses it completes and looks
ordinary. Implementing it would mean writing an integrator-coupled box rescaling, validating that
it reproduces the published relaxation behaviour, and then documenting that its output must not be
used for any fluctuation-derived quantity — work that is not justified when
`MonteCarloBarostat` is available, correct, and already the default.

If Berendsen-style relaxation is wanted for equilibration specifically, the honest route today is
the existing restrained-NPT stages under the Monte Carlo barostat, which relax the box without
claiming a different ensemble.

*Evidence: **DNI**. The method is real and cited; its absence here is an MD-tools policy, not a
statement that the method has no use.*

---

## 11. 2 fs as the baseline; HMR + 4 fs as an explicit option

### 11.1 Why 2 fs is the default

With `constraints=HBonds` the bond stretches involving hydrogen are removed, and the fastest
remaining motions are the bond-angle vibrations involving hydrogen, with periods around 10 fs. A 2 fs
step resolves those with a comfortable margin and is the long-standing default across the
AMBER-family literature, including every validation run in the ff14SB paper [1].

Crucially, at 2 fs **the masses are the ones the force field assigned**. Nothing about the dynamics
is altered to buy the timestep. For a repository whose purpose is method development — comparing
protocols, not maximising throughput — that is the right trade, and it means a trajectory produced
by the defaults can be used for kinetic analysis without a caveat about its inertia.

### 11.2 What is constrained, and what solves the constraints

`constraints: HBonds` is the default, and it is worth being exact about its scope, because the
name is easy to read as more than it is.

| | constrained? |
|---|---|
| bond **lengths** to hydrogen (X–H) | **yes**, every one of them |
| bond lengths between heavy atoms | no |
| bond **angles**, including H–X–H | **no**, never |
| water geometry | rigid when there IS water — see below |

**No angle is ever constrained by this option.** OpenMM's `HAngles` would do that; it is not
offered here, and the schema's enum is `HBonds | AllBonds | None`. That is deliberate and it is
what §11.1 above depends on: with the X–H stretches frozen, the fastest *remaining* motions are
the hydrogen bond-angle vibrations at roughly 10 fs, and 2 fs resolves them with margin. Freezing
those angles too would change which motions the timestep has to resolve, and would change the
ensemble.

A worked example, from a real build (79-atom macrocycle, 38 hydrogens, implicit solvent):

```
constraints                       38   (one per hydrogen, all involving H)
constraint lengths                0.102 - 0.109 nm
HarmonicAngleForce terms         141   of which 85 involve a hydrogen -- all still flexible
```

#### The algorithms are OpenMM's, and are not selectable

MD-tools decides **which** bonds are constrained. It does not decide, and cannot decide, **how**
the constraints are satisfied — OpenMM chooses that per platform from the constraint topology:

- **CCMA** (Constraint Matrix Approximation) solves the general case, and is what handles every
  X–H constraint in an ordinary solute;
- **SETTLE** is the closed-form solver, used for rigid three-site clusters — in practice water.
  It applies only where the cluster really is a rigid triangle, which needs three constraints
  (both O–H bonds *and* the H–H distance). A CH2 or NH2 group under `HBonds` has only its two
  X–H bonds constrained, not the H···H distance, so it is not a rigid triangle and CCMA handles
  it.

**There is no SHAKE.** OpenMM does not implement it; CCMA does the same job by a different
method. A protocol note claiming "SHAKE" is describing a different engine.

Both algorithms are present in the CUDA platform as well as the reference implementation, and
neither is exposed as a setting. **No configuration key selects them, and none is offered**: a
knob that cannot change anything is worse than no knob, because it invites a run record that
claims a choice nobody made.

#### Under implicit solvent there is no SETTLE

`rigid_water` defaults to `true` but is **forced to `false`** under implicit solvent, and
recorded as such — there is no water to hold rigid. So an implicit-solvent run uses CCMA and
nothing else. The build log states the resolved value rather than the requested one.

#### mbondi3 on a solute with no residue names

`mbondi3` is `mbondi2` plus two side-chain corrections that ParmEd selects by RESIDUE and ATOM
NAME: `OD*`/`OE*` in `GLU/ASP/GL4/AS4` to 1.4 A, and `HH*`/`HE*` in `ARG` to 1.17 A. A solute
built from SMILES is one residue with one invented name, so those rules match nothing and the
radii are exactly mbondi2 -- while the build still reports `mbondi3`.

`solute.kind: peptide-like` closes that gap for a head-to-tail cyclic peptide of canonical
residues: the same corrections are applied from a validated chemistry map, after the mbondi2
baseline and before `createSystem`, and the build records the corrected atoms with their old and
new intrinsic radii. `kind: ligand` is unchanged and still reduces to mbondi2, which the build
log continues to state.

See `docs/migration/solute-kind.md` and
`docs/integration/rgdfv-peptide-like-mbondi3.md`.

#### Both routes honour the setting

`constraints.type` reaches the System on the explicit and the implicit route alike. It did not
always: the implicit builder passed `app.HBonds` unconditionally, so the setting was accepted,
echoed back and never applied. Measured on the same input structure, before and after:

| build | constraints | of which heavy-heavy |
|---|---|---|
| `HBonds` + TIP3P | 1782 | 0 |
| `AllBonds` + TIP3P | 1791 | **9** |
| `HBonds` + GBn2 | 12 | 0 |
| `AllBonds` + GBn2, as it was | 12 | **0** — identical to `HBonds` |
| `AllBonds` + GBn2, now | **21** | **9** |

The sharp part was never the ignored setting but the record: the build log stated
`constraints.type  AllBonds  (set)` for a System that had `HBonds`. A provenance record that names
a setting which did not apply is worse than one that omits it.

Both routes now resolve the string through one function, `md_tools.openmm.system.constraint_option`,
so they cannot drift apart again without
`tests/test_constraint_scope.py::test_allbonds_adds_heavy_atom_constraints_under_implicit_solvent`
failing. That test asserts the System; the one beside it asserts that the record and the System
agree, which is the property that was actually broken.

**Two smaller disagreements closed with it.** The schema's enum offers `None` as a string and the
validator compared against Python `None`, so the advertised value was refused as "not supported".
The validator accepted `HAngles`, which the enum does not offer and which this section says is
deliberately unavailable. The enum was the intended policy in both cases and the validator now
matches it.

#### Constraint tolerance

The tolerance is the relative accuracy the solver works to, and it is **not uniform across
protocols today**:

| protocol | constraint tolerance |
|---|---|
| REST2 ladders | **1e-8**, set explicitly (`REST2Protocol.constraint_tolerance`) |
| cMD stages, AIS paths | **1e-5**, OpenMM's integrator default |

Stated rather than quietly reconciled. The ladder is tighter because an exchange compares reduced
potentials computed in different Contexts, and constraint drift enters that comparison directly;
a stage has no such comparison. If a study needs one number across every protocol, that is a
change to make deliberately, not an assumption to carry.

### 11.3 The HMR option

Requested explicitly, never implied by a null:

```yaml
hydrogen_mass_repartitioning:
  enabled: false          # the default
  hydrogen_mass_amu: 3.024
```

This replaced `constraints.hydrogen_mass_amu`, where `null` meant off and a number meant on. One
field carrying both the switch and the value could not express "on, at the default mass", and
reading a file required knowing the convention. The old key now raises a migration error naming
its replacement rather than a generic unknown-key suggestion.

Repartitioning takes mass **from the bonded heavy atom**, so the total is conserved; water is
never repartitioned, because rigid water's hydrogen masses do not limit the timestep; and X-H
constraints are required, because without them the stretch that HMR exists to slow is still the
fastest motion in the system. All three are verified against the built System and recorded.


The complete change is: hydrogen mass 3.024 amu, timestep 4 fs, `constraints: HBonds`,
`rigid_water: true` — set `hydrogen_mass_repartitioning.enabled: true` in a `build-top`
configuration and `dynamics.timestep_fs: 4` in the `build-md` one. (This named
`docs/examples/hmr-4fs.yaml`, which does not exist; the values are stated here instead.) Repartitioning moves mass from each heavy atom onto the
hydrogens bonded to it, lowering the hydrogen angle-bend frequencies and permitting a longer step
[30]. Hopkins et al. showed that with a hydrogen mass around 3 amu, 4 fs integration is
stable for biomolecular systems and reproduces equilibrium structural and thermodynamic properties
[31]. The OpenFE protocol runs 4 fs with 3.0 amu hydrogens as its production default
[6], and the industrial RBFE assessment [7] was run on that
protocol — so the setting has genuine large-scale support **for equilibrium free-energy work**.

### 11.4 The four things that must be kept apart

1. **Stability at 4 fs.** Well supported [30, 31] and, for suitable
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

### 11.5 What the implementation enforces

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

### 11.6 `timestep_fs: auto`, and why the System decides

The default is now the word `auto` rather than the number 2.0.

`build-md` writes run scripts from a configuration file. It never opens `built.xml`, so it cannot
know whether hydrogen mass repartitioning was applied — and the configuration is not evidence: the
two files are written by different commands at different times, and only one of them contains
masses. Under `auto` the decision is deferred to the moment the System is loaded:

| requested | System | outcome |
|---|---|---|
| `auto` | ordinary hydrogens | 2 fs |
| `auto` | repartitioned | 4 fs |
| explicit ≤ 3 fs | either | honoured — 2 fs on an HMR System is slower than necessary, not wrong |
| explicit > 3 fs | repartitioned | honoured |
| explicit > 3 fs | ordinary hydrogens | **refused, before anything integrates** |

The same rule serves cMD, REST2 and AIS, and every log records the resolved value together
with its basis — `ordinary_masses`, `hmr_masses` or `explicit` — and the heaviest hydrogen mass the
decision was made from. A number without its basis cannot be audited: 2 fs chosen by `auto` on an
ordinary System and 2 fs written by hand on a repartitioned one are different decisions.

Step counts remain authoritative throughout. `build-md` prints step counts and, under `auto`, no
picosecond figure at all, because it would be a guess; the derived duration appears in the stage
log once the timestep is known.

*Evidence: **ID**. The 2 fs and 4 fs values themselves are §11.1 and §11.2; this section is about
which of them is applied and on what evidence.*

---

## 12. What these citations do not establish

Read this section before quoting anything above as support for a result.

1. **No citation validates the specific triple ff14SB + Sage 2.2.1 + TIP3P.** The Sage
   protein–ligand benchmark used `ff99SB*-ILDN` [5]. The ff14SB + TIP3P + OpenFF
   workflow evidence comes from software defaults [6] and assessments run on
   them [7, 8], one of which is a preprint and one of which is a
   dataset. The step from "Sage 2.x with an AMBER protein force field in TIP3P" to "Sage 2.2.1 with
   ff14SB in TIP3P" is an extrapolation.
2. **No citation validates ff19SB + Sage 2.2.1 + OPC.** ff19SB + OPC is jointly recommended
   [2]; Sage with OPC is not benchmarked anywhere, and OPC was not in Sage's training
   data. The alternative is offered as two consistent halves, not as a validated whole.
3. **TIP3P is not endorsed as a water model.** §3.2 argues only that it is the water in which
   ff14SB was corrected and Sage's aqueous properties were fit. Its bulk properties are worse than
   OPC's [4], and ff14SB's agreement with experiment depends partly on cancellation with
   that fact [2].
4. **GBn2 is not validated for non-peptidic chemistry.** The GBn2 paper says so
   [15], and §6 measures the consequence on the built System. The implicit ligand route
   is labelled experimental in the record and in this document, and no `igb=8` parity is claimed for
   it.
5. **1.5 nm padding is not a guarantee.** It is common practice in protein–ligand production work
   [9] for solutes that stay folded. Nothing about it bounds the self-interaction of a
   conformation that has not happened yet [18].
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
   `configs/sys/build-top.config`, before `build-top`.
4. **Switch to ff19SB + OPC when the water model is the variable**, or when the science is a
   conformational ensemble of a disordered or marginally stable peptide — and switch every arm.
5. **Archive `inputs/` with the trajectories.** `system.xml` records what was built; the
   configuration records only what was requested. An archive holding trajectories alone cannot say
   which Hamiltonian produced them.
6. **Take the reference topology; do not rebuild an equivalent one.** Every arm should load the
   SAME `system.xml` and topology — the ones the reference simulations were run from, kept with
   the reference data rather than regenerated per arm. Rebuilding "the same" system is where §11
   and backlog entry 6 bite: on the ligand and peptide-like routes, AM1-BCC charges differ between
   two builds of the same input, so two arms that each built their own would differ in the
   Hamiltonian before the methods differed at all. Protein routes assign library charges and are
   deterministic, but taking the built System is the cheaper habit either way — it removes the
   question instead of answering it per project.

   Reference data for shared systems belongs under `$MD_DATA/common/`, so that every project
   comparing against it loads one topology rather than a copy per repository. A copy is a chance
   to diverge; a shared path is not.

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
| thermostat, friction, timestep | requested: `resolved.config → dynamics.{timestep_fs, temperature_K, friction_per_ps}`. Applied: each stage log's *Resolved settings*, where the numerical timestep is the one decided against the masses in the built System rather than the one requested |
| barostat pressure, frequency in steps and in ps, which stages it is active in | requested: `resolved.config → dynamics.pressure_bar`, `dynamics.barostat_interval_steps`. Applied, measured on the built System: the stage log `<stage>.log → barostats.{in_system, active, frequency_steps, interval_ps}` |
| every package version that could change a parameter | `inputs/forcefield.json → package_versions`; and each run's log record → `environment.packages`, which is what registration reads to bind a dataset to the engine that produced it |
| what was built, as one summary beside the lineage hashes | `inputs/forcefield.json` — that file IS the summary of how the System was parameterised; there is no separate one |

`build-top` additionally keeps the user's original input byte-for-byte under `original_inputs/`, keeps
the construction artifacts that carry science under `preparation/`, and writes `SHA256SUMS` over the
whole directory. `build-md` records the parent system's hashes, the stage plan, every seed and
`generated-files.sha256`.

**0.3.x data is not retrofitted.** The defaults changed in 0.4; what 0.3.x ran did not. A
recorded ff19SB/OPC identity is preserved exactly and a missing value is left as `unknown` rather
than filled in from the current default. (This named a retrospective tool
`scripts/retrofit_fair_v030.py`, which does not exist in this repository.)

---

## 15. Sources

Numbered in order of first citation, and cited by number in the text above. `docs/scientific-defaults.bib` holds the same entries keyed by name, with the DOI of each verified against the
Crossref REST API on 2026-08-27.

Entries that are software documentation, a dataset or a preprint are labelled as such here and
in §2. None of them is ever used to support a scientific claim a primary source could support
instead.

[1] J. A. Maier; C. Martinez; K. Kasavajhala; L. Wickstrom; K. E. Hauser; C. Simmerling. ff14SB:
Improving the Accuracy of Protein Side Chain and Backbone Parameters from ff99SB. J. Chem. Theory
Comput., 2015, 11, 3696–3713.

[2] C. Tian; K. Kasavajhala; K. A. A. Belfon; L. Raguette; H. Huang; A. N. Migues; J. Bickel;
Y. Wang; J. Pincay; Q. Wu; C. Simmerling. ff19SB: Amino-Acid-Specific Protein Backbone Parameters
Trained against Quantum Mechanics Energy Surfaces in Solution. J. Chem. Theory Comput., 2020, 16,
528–552.

[3] W. L. Jorgensen; J. Chandrasekhar; J. D. Madura; R. W. Impey; M. L. Klein. Comparison of simple
potential functions for simulating liquid water. J. Chem. Phys., 1983, 79, 926–935.

[4] S. Izadi; R. Anandakrishnan; A. V. Onufriev. Building Water Models: A Different Approach.
J. Phys. Chem. Lett., 2014, 5, 3863–3871.

[5] S. Boothroyd; P. K. Behara; O. C. Madin; D. F. Hahn; H. Jang; V. Gapsys; J. R. Wagner;
J. T. Horton; D. L. Dotson; M. W. Thompson; J. Maat; T. Gokey; L.-P. Wang; D. J. Cole; M. K. Gilson;
J. D. Chodera; C. I. Bayly; M. R. Shirts; D. L. Mobley. Development and Benchmarking of Open Force
Field 2.0.0: The Sage Small Molecule Force Field. J. Chem. Theory Comput., 2023, 19, 3251–3275.

[6] The Open Free Energy Consortium. OpenMM Relative Free Energy Protocol — default settings.
https://docs.openfree.energy/en/v0.15.0/reference/api/openmm_rfe.html, 2024. *Software
documentation.*

[7] H. M. Baumann; J. T. Horton; M. M. Henry; A. Travitz; B. Ries; R. J. Gowers; D. W. H. Swenson;
I. Pulido; D. Rufa; D. L. Dotson; et al. Large-scale collaborative assessment of binding free energy
calculations for drug discovery using OpenFE. ChemRxiv preprint, posted 2025-12-18. *Preprint, not
peer reviewed.*

[8] I. Alibay. OpenFF 2.2.1 Relative Binding Free Energy Benchmarks. Zenodo dataset, version v2,
2025-10-21. *Dataset, not a peer-reviewed publication.*

[9] D. F. Hahn; V. Gapsys; B. L. de Groot; D. L. Mobley; G. Tresadern. Current State of Open Source
Force Fields in Protein-Ligand Binding Affinity Predictions. J. Chem. Inf. Model., 2024, 64,
5063–5076.

[10] The Open Force Field Initiative. openff-forcefields releases: Sage 2.2.0 (tag 2024.04.0) and
Sage 2.2.1 (tag 2024.09.0). https://github.com/openforcefield/openff-forcefields/releases, 2024.
*Software documentation.*

[11] J. Wang; W. Wang; P. A. Kollman; D. A. Case. Automatic atom type and bond type perception in
molecular mechanical calculations. J. Mol. Graphics Modell., 2006, 25, 247–260.

[12] A. Jakalian; D. B. Jack; C. I. Bayly. Fast, efficient generation of high-quality atomic
charges. AM1-BCC model: II. Parameterization and validation. J. Comput. Chem., 2002, 23, 1623–1641.

[13] J. Wang; R. M. Wolf; J. W. Caldwell; P. A. Kollman; D. A. Case. Development and testing of a
general amber force field. J. Comput. Chem., 2004, 25, 1157–1174.

[14] The Amber developers. GAFF2 parameters, distributed with AmberTools as gaff2.dat.
https://ambermd.org/AmberModels_organic.php, 2026. *Software documentation.*

[15] H. Nguyen; D. R. Roe; C. Simmerling. Improved Generalized Born Solvent Model Parameters for
Protein Simulations. J. Chem. Theory Comput., 2013, 9, 2020–2034.

[16] J. Mongan; C. Simmerling; J. A. McCammon; D. A. Case; A. Onufriev. Generalized Born Model with
a Simple, Robust Molecular Volume Correction. J. Chem. Theory Comput., 2007, 3, 156–169.

[17] P. H. Hünenberger; J. A. McCammon. Ewald artifacts in computer simulations of ionic solvation
and ion–ion interaction: A continuum electrostatics study. J. Chem. Phys., 1999, 110, 1856–1872.

[18] K. El Hage; F. Hédin; P. K. Gupta; M. Meuwly; M. Karplus. Valid molecular dynamics simulations
of human hemoglobin require a surprisingly large box size. eLife, 2018, 7, e35560.

[19] T. Darden; D. York; L. Pedersen. Particle mesh Ewald: An N·log(N) method for Ewald sums in
large systems. J. Chem. Phys., 1993, 98, 10089–10092.

[20] U. Essmann; L. Perera; M. L. Berkowitz; T. Darden; H. Lee; L. G. Pedersen. A smooth particle
mesh Ewald method. J. Chem. Phys., 1995, 103, 8577–8593.

[21] Z. Zhang; X. Liu; K. Yan; M. E. Tuckerman; J. Liu. Unified Efficient Thermostat Scheme for the
Canonical Ensemble with Holonomic or Isokinetic Constraints via Molecular Dynamics. J. Phys. Chem.
A, 2019, 123, 6056–6079.

[22] B. Leimkuhler; C. Matthews. Efficient molecular dynamics using geodesic integration and
solvent–solute splitting. Proc. R. Soc. A, 2016, 472, 20160138.

[23] B. Leimkuhler; C. Matthews. Rational Construction of Stochastic Numerical Methods for Molecular
Sampling. Appl. Math. Res. eXpress, 2013, 2013, 34–56.

[24] B. Leimkuhler; C. Matthews. Robust and efficient configurational molecular sampling via
Langevin dynamics. J. Chem. Phys., 2013, 138, 174102.

[25] K.-H. Chow; D. M. Ferguson. Isothermal-isobaric molecular dynamics simulations with Monte Carlo
volume sampling. Comput. Phys. Commun., 1995, 91, 283–289.

[26] J. Åqvist; P. Wennerström; M. Nervall; S. Bjelic; B. O. Brandsdal. Molecular dynamics
simulations of water and biomolecules with a Monte Carlo constant pressure algorithm. Chem. Phys.
Lett., 2004, 384, 288–294.

[27] L. Wang; R. A. Friesner; B. J. Berne. Replica Exchange with Solute Scaling: A More Efficient
Version of Replica Exchange with Solute Tempering (REST2). J. Phys. Chem. B, 2011, 115, 9431–9438.

[28] H. J. C. Berendsen; J. P. M. Postma; W. F. van Gunsteren; A. DiNola; J. R. Haak. Molecular
dynamics with coupling to an external bath. J. Chem. Phys., 1984, 81, 3684–3690.

[29] M. Bernetti; G. Bussi. Pressure control using stochastic cell rescaling. J. Chem. Phys., 2020,
153, 114107.

[30] K. A. Feenstra; B. Hess; H. J. C. Berendsen. Improving efficiency of large time-scale molecular
dynamics simulations of hydrogen-rich systems. J. Comput. Chem., 1999, 20, 786–798.

[31] C. W. Hopkins; S. Le Grand; R. C. Walker; A. E. Roitberg. Long-Time-Step Molecular Dynamics
through Hydrogen Mass Repartitioning. J. Chem. Theory Comput., 2015, 11, 1864–1874.

**Not cited by number above.** Present in `docs/scientific-defaults.bib` and part of this
document's source set, but no passage above carries an inline marker for them. ParmEd is
discussed by name in §5 without one; the nonequilibrium-work entries belong to AIS, which
this document mentions only in passing.

[32] P. Eastman; J. Swails; J. D. Chodera; R. T. McGibbon; Y. Zhao; K. A. Beauchamp; L.-P. Wang;
A. C. Simmonett; M. P. Harrigan; C. D. Stern; R. P. Wiewiora; B. R. Brooks; V. S. Pande. OpenMM 7:
Rapid development of high performance algorithms for molecular dynamics. PLoS Comput. Biol., 2017,
13, e1005659.

[33] P. Eastman; R. Galvelis; R. P. Peláez; C. R. A. Abreu; S. E. Farr; E. Gallicchio; A. Gorenko;
M. M. Henry; F. Hu; J. Huang; A. Krämer; J. Michel; J. A. Mitchell; V. S. Pande;
J. P. G. L. M. Rodrigues; J. Rodriguez-Guerra; A. C. Simmonett; S. Singh; J. Swails; P. Turner;
Y. Wang; I. Zhang; J. D. Chodera; G. De Fabritiis; T. E. Markland. OpenMM 8: Molecular Dynamics
Simulation with Machine Learning Potentials. J. Phys. Chem. B, 2024, 128, 109–116.

[34] The OpenMM Development Team. OpenMM Python API: MonteCarloBarostat, LangevinMiddleIntegrator.
http://docs.openmm.org/latest/api-python/, 2026. *Software documentation.*

[35] M. R. Shirts; C. Klein; J. M. Swails; J. Yin; M. K. Gilson; D. L. Mobley; D. A. Case;
E. D. Zhong. Lessons learned from comparing molecular dynamics engines on the SAMPL5 dataset.
J. Comput.-Aided Mol. Des., 2017, 31, 147–161.

[36] C. Jarzynski. Nonequilibrium Equality for Free Energy Differences. Phys. Rev. Lett., 1997, 78,
2690–2693.

[37] R. M. Neal. Annealed importance sampling. Stat. Comput., 2001, 11, 125–139.