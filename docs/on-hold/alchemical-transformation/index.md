# Alchemical transformation — on hold

!!! warning "On hold since 2026-10-13"
    MD-tools is a conformational-sampling toolkit for now. Alchemical transformation is **not part
    of the current release**: its tutorials are not taught from the main tutorial list, and it is not
    re-validated or tested for a release. Everything below was last run against **0.6.4**. See
    [On hold](../index.md) for what else is waiting.

## What exists

| part | what it is | last run against |
|---|---|---|
| [Alchemical topology](method/README.md) | `md-openmm combine-topology`: two endpoint Systems over one index space, and the `protocol: alchemical` ladder `build-md` generates from it | 0.6.4 |
| [What the runtime computes](method/validation.md) | the evidence: a nine-leg hydration campaign checked against experiment and against a second alchemical route | 0.6.4 |
| [Ethanol: hydration free energy](tutorial/ethanol/index.md) | the smallest alchemical free energy there is, with an experimental number to check it against | 0.6.4 |
| [Ethane → chloroethane: relative hydration free energy](tutorial/ethane-chloroethane/index.md) | one atom apart: the relative-free-energy pattern a protein–ligand campaign uses | 0.6.4 |

## What is implemented, and what is not

Implemented: a plan with four modes (`single`, `hybrid`, `dual`, `decoupling`), the five λ
components named as OpenFE names them, an Amber18 softcore, and a generated ladder that `md-run`
dispatches. Not implemented, and refused by name: `lambda_path: staged`, `mode: separated`, a
decoupling of a charged ligand, and atom mapping across a ring. The detail is on the
[method page](method/README.md).

## What would take it off hold

1. A decision that alchemical work is back in scope.
2. The three failing CUDA tests in `test_alchemy_hamiltonian_cuda.py` explained: they fail the same
   way on `main`, deterministically and without MPS, by 0.0063 kJ/mol against a tolerance of 0.0021
   on one of them.
3. The two tutorials re-run on the then-current release, because every number on them was produced
   by 0.6.4.
4. The development plans — [FEP–REST2 and relatives](../../development/0.7.1/AIMS.md) and
   [water sampling](../../development/0.7.2/AIMS.md) — are separate and come after.
