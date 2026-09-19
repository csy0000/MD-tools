# Development

**Nothing in this directory describes the installed package.** It describes work that is planned
or under construction, on branches that have not been released. What the package does today is in
the [documentation index](../README.md).

Four branches were opened on 2026-09-18 from `e524e0e` on the 0.6.0 development line (released as `v0.6.0` at `3927105`; the `dev-0.6.0` branch has since been deleted), following the
[20260918 parallel development instruction](../claudecode-instructions/20260918_parallel-0.6.1-0.7.x-development.md).
That instruction is the shared roadmap; the aims pages below specialize it per branch.

| branch | what it is for | state |
|---|---|---|
| [0.6.1](0.6.1/AIMS.md) | selective explicit-solvent REST2: AMBER residue masks for backbone and sidechains, individually chosen ligand instances | under construction |
| [0.7.0](0.7.0/AIMS.md) | conventional alchemy: `md-openmm combine-topology`, Amber18 softcore, TI and FEP, solvation and binding free energies | under construction |
| [0.7.1](0.7.1/AIMS.md) | FEP–REST2, EDS, RE-EDS, ATM | **planning only** — aims and references, no implementation |
| [0.7.2](0.7.2/AIMS.md) | grand-canonical and nonequilibrium water sampling | **planning only** — aims and references, no implementation |

[Shared contracts](shared-contracts.md) holds the records 0.6.1 and 0.7.0 must agree on before
their code diverges: selection identity, the alchemical topology plan, and the naming of
Hamiltonian state coordinates. A change to a contract is one commit that updates the document and
its fixtures, never a private version in a worker branch.

Each branch keeps a `STATUS.md` — baseline and contract commits, milestone states, worker
branches, evidence locations and blockers — and a `handoffs/` directory holding one report per
worker session. Workers own their own handoff; the coordinator owns the status files, the root
indexes and the backlog.

## Release ancestry

Opening four branches now does not mean releasing or implementing them together.

- 0.7.0 must contain the tested 0.6.1 integration commit and any accepted 0.6.0 fixes.
- 0.7.1 incorporates a completed 0.7.0 before any dependent method is implemented or released.
- 0.7.2 incorporates the relevant 0.7.1 foundation before its combined-method validation.

0.6.0 itself is still under the user's testing. No tag, package publication or merge to `main` is
authorized for any of this.

## What a milestone needs before it is called done

An implemented artifact, an executable example, and evidence: a named fixture, an independent
reference, a stated tolerance fixed *before* the comparison, the platform it ran on, and the
command. A skipped lane is not a pass, a CPU run is not CUDA evidence, and a green unit-test suite
is not proof that a free-energy method is correct. See
[promoting a method](../promoting-a-method.md), whose requirements apply to everything here.
