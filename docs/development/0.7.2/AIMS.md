# 0.7.2 — grand-canonical water sampling

**Planning only.** This branch carries aims and references. No implementation is authorized on it
during the 0.6.1 / 0.7.0 wave, and nothing here is available in any release.

Baseline: `e524e0e` on the 0.6.0 development line (released as `v0.6.0` at `3927105`; the `dev-0.6.0` branch has since been deleted). Prerequisite: the relevant 0.7.1 foundation, because the
combined-method validation is where this feature earns its place — water sampling that is correct
on its own but wrong inside an alchemical cycle is worse than not having it.

## Purpose

Insert and delete water molecules in a chosen region under a fixed chemical potential, by
conventional Monte Carlo and by nonequilibrium (work-based) moves, and then combine that with
validated alchemical and REST2 protocols. Buried waters that cannot equilibrate by diffusion are
the reason: a binding free energy computed with the wrong pocket occupancy is precise and wrong.

Initial scope is **water**. Broader ligand insertion and deletion is a later extension.

## What must be specified before anything is implemented

A GCMC implementation is a set of conventions, and every one of them changes the answer:

- the chemical potential and the standard state it is referred to, with the direction of every sign;
- the sampling region and its volume, and what happens to a molecule that leaves it;
- proposal probabilities for insertion and deletion, and the indistinguishability factor (`1/N!`
  bookkeeping) that makes acceptance correct;
- ghost-particle identity: which particles are non-interacting, how the count of occupied
  particles is tracked, and how that count enters acceptance;
- periodic boundaries, electrostatics (PME with a changing particle count is not free), and the
  treatment of momenta on an accepted insertion;
- a transactional accepted/rejected move: a rejected move must restore the previous state exactly,
  including the RNG stream, and a crash mid-move must not leave a half-inserted molecule.

For nonequilibrium moves, additionally: the protocol work definition, the reversibility of the
integrator and the proposal, and any path-probability or shadow-work term the acceptance needs.

**Reusing the AIS trajectory generator does not establish a correct Monte Carlo acceptance rule.**
AIS measures work along a switching path; GCMC needs that work inside an acceptance criterion with
its own detailed-balance argument. The two share machinery, not a proof.

## Validation aims

| fixture | quantity | reference |
|---|---|---|
| ideal gas in a box | the number distribution | the analytic grand-canonical law |
| bulk water, matched water model | chemical potential and density | the published reference for that model |
| any move | rejected-move state restoration | bitwise comparison with the pre-move state |
| a buried pocket | occupancy | convergence from several different initial occupancies to the same answer |
| an alchemical cycle with water sampling | free energy | the same cycle with fixed waters, with the ensemble bookkeeping accounted for |

Conventional and nonequilibrium methods must agree within statistical uncertainty wherever both
are converged. Disagreement is a defect in one of them, not a choice between them.

## Non-goals

- Implementation during the 0.6.1 / 0.7.0 wave.
- Ligand or ion insertion and deletion in this milestone.
- Adopting any reference implementation's conventions without re-deriving them here.

## References

R13 (`deGrootLab/GrandFEP`) is a reference for how alchemy, REST2 and water sampling are
integrated — not proof that its choices transfer unchanged. R14 (`essex-lab/grand`) covers
conventional GCMC and nonequilibrium water moves; follow its primary method papers
(<https://doi.org/10.1021/acs.jcim.0c00648>, <https://doi.org/10.1021/acs.jctc.2c00823>) rather
than the code alone. The full register is section 10 of the
[20260918 instruction](../../claudecode-instructions/20260918_parallel-0.6.1-0.7.x-development.md).
