# 0.7.1 — advanced alchemical sampling

**Planning only.** This branch carries aims and references. No implementation is authorized on it
during the 0.6.1 / 0.7.0 wave, and nothing here is available in any release.

Baseline: `dev-0.6.0` at `e524e0e`. Prerequisite: a completed and integrated 0.7.0 — every
milestone below composes the alchemical Hamiltonian 0.7.0 defines, and none of them can be
validated against a Hamiltonian that does not yet exist.

## Purpose

Four methods that sample an alchemical space more efficiently than fixed-lambda windows do, each
a separately validated milestone with its own acceptance evidence. They share the runtime and the
evaluation interfaces; they do not share a state coordinate, and there is no requirement to force
EDS or ATM into a lambda interpolation to make them look alike.

## Milestones

### 1. FEP–REST2

Compose alchemical coupling with the selective REST2 of 0.6.1 in ONE defined Hamiltonian, with a
term-by-term derivation. Specify the exchange topology (which pairs may swap, over which
coordinate) and the cross-state evaluations each exchange needs.

Two reductions are acceptance criteria, not remarks:
- at `tau = 0` it must reduce exactly to the 0.7.0 alchemical Hamiltonian;
- at the physical alchemical endpoints it must reduce exactly to the intended REST2 state.

A generic scaler multiplying arbitrary new custom forces by `(1-tau)` is not an implementation of
this. The alchemical softcore terms are not ordinary nonbonded terms, and what scaling them means
has to be derived before it is coded.

`tau` and the alchemical lambdas are different named coordinates and must never share a parameter
name or be compared as numbers — see [shared contracts](../shared-contracts.md).

### 2. EDS

Enveloping distribution sampling: evaluate several endpoints, combine them with a numerically
stable log-sum-exp enveloping potential, and support energy offsets and a smoothing parameter.
Implement the derivatives and endpoint reweighting.

Validate toy energies, forces and derivatives against independent arithmetic — a hand-written
two-state expression evaluated in plain Python — before any molecular simulation. A log-sum-exp
that overflows or silently loses a state is invisible in a trajectory.

### 3. RE-EDS

Replica exchange over EDS states, built only on a validated EDS. Adds offset and smoothing
optimization, exchanges through the existing MPI authority, sampling diagnostics, and molecular
free-energy validation.

The linked `rinikerlab/reeds` workflow is GROMOS-based. Its OpenMM study uses a shifted
reaction field with an atom-based cutoff: **reaction-field evidence is not PME evidence**, and
importing a pipeline does not supply a validated PME OpenMM implementation.

### 4. ATM

Prefer OpenMM's native `ATMForce` where it is supported and suitable; use the
`openmm-atmmetaforce-plugin` as a reference rather than a dependency. Validate, for each
application claimed: the transfer itself, displacement geometry, periodic boundaries, restraints,
softcore mapping, state energies, and the free-energy corrections that make the result a binding
or transfer free energy rather than a number.

## Non-goals

- Implementation during the 0.6.1 / 0.7.0 wave.
- Any claim of support for a method whose reductions and corrections have not been demonstrated.
- Replacing the 0.7.0 fixed-lambda path; these methods are additions to it.

## References

R1 (REST2), R4–R9 (Amber18, OpenFE, PyMBAR, OpenMM forces), R10 (EDS / RE-EDS), R11 (ATM plugin),
R12 (native `ATMForce`) — the register is section 10 of the
[20260918 instruction](../../claudecode-instructions/20260918_parallel-0.6.1-0.7.x-development.md).
Pin versions and commits when implementation begins, not now.

## Acceptance criteria (to be refined when the branch becomes active)

Each milestone needs, independently of the others: analytic agreement on a toy system; the stated
reductions demonstrated numerically; finite-difference derivative checks at several step sizes;
a free-energy result on a small fixture agreeing with an independent reference within a tolerance
fixed before the comparison; and CUDA evidence from a lane that actually ran with the feature on.
