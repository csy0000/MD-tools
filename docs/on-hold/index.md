# On hold

Work that was started, or planned, and is **deliberately not being developed now**. Nothing on these
pages is part of the current release's promise: it is not re-validated, its tests are not part of
the release gate, and the tutorials do not teach it. It is kept, in the tree and in git history, so
it can be picked up again without being rebuilt.

**The focus for now** is conformational sampling — cMD, REST2 and AIS — and the preparation of the
small molecules they run on: Open Babel protonation at a stated pH, ligand amide scaling in REST2,
and BespokeFit torsion refinement (`solute.torsion_refinement`). *Decided 2026-10-13.*

| what | state | where |
|---|---|---|
| **Alchemical transformation** — `md-openmm combine-topology`, the `alchemical` protocol, hydration and relative free energies, and the two tutorials that were run on it | built and run in 0.6.4, **on hold** | [Alchemical transformation](alchemical-transformation/index.md) |
| **Advanced alchemical sampling** — FEP–REST2, EDS, RE-EDS, ATM | planning only; no implementation | [aims](../development/0.7.1/AIMS.md) |
| **Grand-canonical and nonequilibrium water sampling** | planning only; no implementation | [aims](../development/0.7.2/AIMS.md) |
| **The protein–ligand campaign** — one TYK2 system with three congeneric ligands, shared by the methods above | planned; nothing measured | [plan](../development/protein-ligand-campaign.md) |
| **Umbrella sampling** — its release is pushed to a later one; the method page stays under Methods and its tutorial page is not written | postponed, not abandoned | [method page](../openmm_methods/umbrella/README.md) |

## What "on hold" means for each kind of thing

* **Documentation** is kept and builds with the rest of the site, under a banner that says it is on
  hold.
* **Code and tests** stay in the tree. They are not deleted and not reported as failing; they are
  also not part of the release gate. Three CUDA tests in `test_alchemy_hamiltonian_cuda.py` fail
  both on `main` and on this branch, and are known and unexplained.
* **Nothing here is promised to a user** until it is taken off hold, re-run on the current release
  and given its tests back.
