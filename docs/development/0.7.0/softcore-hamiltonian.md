# The Amber18 softcore Hamiltonian (0.7.0, milestone A2)

**Under construction, not in any release.** Owner: S3. Code: `md_tools.alchemy.softcore`,
`md_tools.alchemy.hamiltonian`. Evidence and status: [handoffs/S3.md](handoffs/S3.md).

## What it is

A Hamiltonian over one hybrid particle set, built from the two end-state Systems of a topology
plan ([topology-plan.md](topology-plan.md)): common particles (C), A-only particles that
**disappear** (present at lambda = 0) and B-only particles that **appear** (present at lambda = 1).

```text
U(le, ls, lb) = (1 - le) El_A(le) + le El_B(le)        electrostatics
              + (1 - ls) LJ_A(ls) + ls LJ_B(ls)        Lennard-Jones, dispersion correction included
              + (1 - lb) Bonded_A + lb Bonded_B        bonded
              + U_internal                             inside a softcore region, unscaled
```

Three named components, `lambda_electrostatics`, `lambda_sterics`, `lambda_bonded`. The Amber18
one-step transformation is the diagonal `le = ls = lb = lambda`; a staged path (decharge, then
sterics) is another path over the same Hamiltonian. The functional form does not depend on the
path, and a path is `md_tools.alchemy.paths`' business.

Amber mixes **energies**, not parameters: `(1 - l) q_A,i q_A,j + l q_B,i q_B,j`, which is not
`q_i(l) q_j(l)`. That holds for the common core's changing charges and LJ as well as for softcore.

## The softcore pair (`softcore.py`)

Amber18 manual section 21.1.5 (R5; `Amber18.pdf`, sha256 `81b7dd57…07791`), for a region atom i
and a common atom j:

```text
disappearing (A):  4 eps (1 - l) [ 1/(alpha l + (r/sigma)^6)^2 - 1/(alpha l + (r/sigma)^6) ]     (21.5)
appearing (B):     4 eps l       [ 1/(alpha (1-l) + (r/sigma)^6)^2 - ... ]                         (21.6)
electrostatics:    (1 - l) q_i q_j / (4 pi eps0 sqrt(beta l + r^2)),  l <-> 1 - l for B            (21.7)
```

`sc: true`, `scalpha = 0.5`, `scbeta = 12 A^2` (0.12 nm^2) are the MD-tools defaults. They are not
a claim that AMBER's `ifsc` defaults to 1.

**Under PME** the manual gives only the vacuum form. pmemd computes the direct-space pair as
`q_i q_j erfc(kappa r) / sqrt(r^2 + beta l)`: erfc at the true distance, only 1/r softened. The
reciprocal, self and background terms are each end state's own, untouched. The sources are the
pmemd 26 source, `pairs_calc_ti_AUTO.i` (`pairs_calc_ti_sc_common_V0_cut1_mefv`, sha256
`8821629e…`) and `cuda/gti_nonBond_kernels.cu` (eleGauss = 0, eleExp = 2, sha256 `4ebab972…`),
from `pmemd26.tar.bz2` sha256 `0478ccce…23a14`. That is the form implemented. Softening a
real-space term while the pair's original contribution survives elsewhere would give smooth, wrong
free energies. Here the end state's NonbondedForce keeps the pair's reciprocal share, and a custom
force adds the delta `w q q erfc(kappa r) (1/sqrt(r^2 + beta l) - 1/r)`.

**Refused by name:** `gapsys` (OpenFE's default), `beutler` (OpenFE's LJ option), `openfe`,
`smoothstep` and `amber20` (the Amber20+ `gti_lam_sch` / `gti_scale_beta` forms). None of them is
Amber18, and none runs under that name.

`sc: false` is ordinary linear mixing. It is allowed only when nothing appears or disappears, and
is refused otherwise rather than softened behind the user's back.

## Every intramolecular and exception rule

| pair or term | treatment | source |
|---|---|---|
| C-C nonbonded, C-C exception | energy-mixed between end states | eq. 21.3 |
| region-C nonbonded | softcore, eqs. 21.5-21.7, weighted | 21.1.5 |
| region-internal non-excluded pair | unscaled, vacuum Coulomb + LJ, no cutoff; removed from the weighted Ewald sum like an exclusion | 21.1.5 "interactions among the disappearing atoms are not changed"; pmemd `gti_cut = 1` |
| region-internal exception (1-4) | unscaled, from the end state where the region is physical | same |
| region-C 1-4 exception | `sc_boundary_14`: `scaled` mixes it (zero at the dummy end) = pmemd 20+ `gti_add_sc = 1`; `unscaled` keeps it at full strength = Amber18 21.1.5 (`gti_add_sc = 0`) | see below |
| A-only x B-only | excluded at every lambda | plan |
| bonded term on C only | energy-mixed with `lambda_bonded` | eq. 21.3 |
| bonded term on a region, identical in both end states (retained at the dummy end) | unscaled | 21.1.5 |
| bonded term on a region with force constant 0 at its dummy end only (the plan's `dummy-removed`) | mixed with `lambda_bonded` | pmemd 20+ `gti_bat_sc = 1` analogue |
| bonded term on a region differing any other way | refused | — |

**`sc_boundary_14` is the one departure from the literal Amber18 text, and its default is
PROVISIONAL.** The Amber20+ manual calls `gti_add_sc = 0` theoretically incorrect. It also couples
a dummy to physical coordinates through 1-4s, so the plan's single-anchor dummy would no longer
factorise. The default is therefore `scaled`, pending the user's decision (asked through S0,
2026-09-19). Both rules are implemented and tested. The record keeps the softcore form
(`softcore_function: amber18`) and the boundary rule as two separate facts, so no record calls the
combination "amber18".

**Consequence for the end states.** Under `scaled`, `U(0)` is System A and `U(1)` is System B,
plus the other region's internal nonbonded energy. That energy is a function of the group's
internal coordinates only, so the group remains separable, and it is zero for a group with no
internal 1-4 or 1-5+ pair (a single atom; ethanol's O-H). The plan annihilates a dummy's internal
pairs too, so this is an open contract question with S2, recorded in their handoff.

## The dispersion correction

It is each end state's **own** OpenMM correction, mixed linearly in `lambda_sterics`:
`(1 - ls) D_A(V) + ls D_B(V)`. pmemd does the same, weighting each TI region's correction by its
weight, with an unsoftened tail. It cannot be assembled from per-subset corrections. Measured on
OpenMM 8.6 and pinned by `test_openmm_long_range_corrections_are_not_additive_over_subsets`:

- NonbondedForce, and CustomNonbondedForce without groups, return `N/(N+1)` times the tail summed
  over distinct pairs **and self pairs**;
- CustomNonbondedForce **with** interaction groups returns exactly `N/(N+1)` times the
  distinct-pair tail of the group, where N is every particle in the System.

A first construction split the LJ into static, changing and softcore subsets and missed S2's end
states by 2e-4 and 5e-3 kJ/mol; with the correction off it matched to 5e-12. Now D_A and D_B come
from OpenMM's own coefficient formula (checked against OpenMM's number). A single-pair
CustomNonbondedForce carries the part the static NonbondedForce lacks: its pair energy is
identically zero inside the cutoff (`step(r - rc)`), and its long-range correction is the 1/V term,
calibrated at build. Both end states are recovered to < 1e-8 kJ/mol, at the built box and at an
expanded one.

Three further OpenMM 8.6 behaviours this depends on, each measured:

1. A NonbondedForce's dispersion correction is **not** recomputed when an epsilon offset changes
   through a global parameter. So nothing lambda-dependent is left in a NonbondedForce's LJ.
2. A CustomNonbondedForce's long-range correction **is** recomputed on a global-parameter change,
   by numerical quadrature. So the dispersion carrier's energy departs from exact linearity in
   `lambda_sterics` by up to **4.9e-8 kJ/mol** (2e-8 kT; measured over 41 lambdas on S2's
   chloroethane plan), while its derivative is the exact slope (D_B - D_A, to 1e-8). This is
   negligible for any estimator, but it bounds finite-difference checks: the tests difference
   every other group at full precision (1e-10 kJ/mol) and check the carrier's slope on its own.
3. If an energy-parameter derivative of a long-range correction simplifies to zero, OpenMM
   **aborts the process** ("Cannot use long range correction with a force that does not depend on
   r"). It is an abort, not an exception. The carrier therefore names no lambda when D_A = D_B.

A dispersion correction together with a switching function is refused until the switched
integral is reproduced.

## Derivatives

`derivatives(context, state)` returns the complete `dU/d lambda_k` for all three components.
**It never uses `U1 - U0`:** that identity is AIS's, and it is exact only for a linear path. It is
computed per force group:

- custom forces (softcore, changing LJ, bonded mixing, dispersion carrier): OpenMM
  energy-parameter derivatives, each expression written in the public names;
- the two end-state NonbondedForces: exact algebra rather than OpenMM derivatives. Their energy is
  `qscale^2 Q + w_el X + w_lj Y + const` exactly, so Q, X and Y are energy differences at weights 1
  and 0. This takes six group evaluations; the Context's parameters are restored afterwards.

`derivative_components` gives the same numbers split by force group, for per-component error
reports.

## How it is built

| group | force | content |
|---|---|---|
| 0 | as given | every force identical in both end states |
| 1 | NonbondedForce | end state A's charges x sqrt(1 - le): its Ewald energy x (1 - le); LJ common to both end states |
| 2 | NonbondedForce | end state B's charges x sqrt(le) |
| 3 | CustomNonbondedForce | common particles whose LJ changes |
| 4 / 5, 6 / 7 | CustomNonbondedForce | region A / region B: electrostatic delta, softcore LJ |
| 8 | CustomBondForce | region-internal pairs and exceptions, unscaled |
| 9 | Custom{Bond,Angle,Torsion}Force | bonded forces that differ, mixed |
| 10 | CustomNonbondedForce | the dispersion carrier |

The square-root charge weights are **derived** Context parameters (`mdt_alchemy_*`), which is why
a state is set with `set_state(context, state)` and never with `Context.setParameter` on a public
name alone. `context_parameters(state)` is the one definition of every parameter value, public and
derived. A serialised System plus that dict reproduces the energies with no live Context.
Group 16 is kept free for S4's restraint.

Known numerical limitation: the electrostatic delta cancels the NonbondedForce's `q q erfc/r`
analytically. Near an overlap (r -> 0 at lambda > 0) the cancellation is also numerical. That is
exact enough in double precision (tested at r = 1e-5 nm), but costs precision in single-precision
forces.

Cost: two PME reciprocal sums per evaluation, as in pmemd, plus a few custom kernels over the
softcore regions. It has not been measured on a device yet.

## Interface

```python
from md_tools.alchemy.hamiltonian import build_hamiltonian, from_plan
from md_tools.alchemy.softcore import SoftcoreSettings

h = from_plan(plan, settings=SoftcoreSettings())   # or build_hamiltonian(system_a, system_b, a_only, b_only)
h.system                  # an openmm.System
h.parameter_names         # ("lambda_electrostatics", "lambda_sterics", "lambda_bonded")
h.set_state(context, {"lambda_electrostatics": 0.5, "lambda_sterics": 0.5, "lambda_bonded": 0.5})
h.energy(context, state); h.energy_components(context, state)
h.derivatives(context, state); h.derivative_components(context, state)
h.context_parameters(state); h.record     # the provenance record, plain data
```

## Refused

Particles that differ in count, mass or constraints between the end states; virtual sites; a
dummy that carries charge or epsilon, or a nonzero exception; A-only x B-only pairs not excluded; a
net charge change; a common particle with LJ at one end only; any nonbonded force other than one
plain NonbondedForce (GB, custom nonbonded, AMOEBA, Drude, ATM, an already-alchemical System);
nonbonded methods other than NoCutoff and PME; dispersion correction with a switching function; a
differing force that is not a harmonic bond, harmonic angle or periodic torsion; a bonded term on a
softcore particle that differs other than by the plan's dummy removal; `sc: false` with unique
particles.
