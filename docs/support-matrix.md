# Supported versions and what portability means

## Matrix

Determined by what actually passes, not by aspiration. A version enters this table when a CI run
gates it, and is described as *locally verified* until then. The only workflow is
`release-validation`, which runs on `workflow_dispatch` and on an `openmm-v*` tag — not on every
push — so "gated" here means gated at release, not continuously.

| | version | status |
|---|---|---|
| Python | 3.12 | pinned by `environment-ci.yml`; gated by `release-validation`; locally verified on 3.12.13 |
| OpenMM | 8.6.0 | pinned by `environment-ci.yml`; gated by `release-validation`; acceptance run on the conda-forge **release** package `openmm 8.6.0 py312hdfcc665_0`, which reports `openmm.version.version` as `8.6.0.dev-c6173db` — the release identity comes from the package, not that string |
| MDTraj | 1.11.1 locally | reads the AIS source DCD and its box vectors; installed and import-checked |
| OpenFF toolkit | 0.19.0 locally | installed from the conda environment; unpinned in the solve |
| openmmforcefields | 0.16.0 locally | as above |
| AmberTools | `sqm`, `antechamber`, `tleap` on PATH | presence and AM1-BCC readiness checked at install and in CI |
| ParmEd / RDKit | 4.3.1 / 2026.03.1 locally | installed and import-checked |
| OpenFF force fields | `openff-2.2.1` (Sage 2.2.1) | shipped by `openforcefields`; loaded and asserted by a non-GPU test |
| OpenMM force fields | `amber14-all.xml` + `amber14/tip3p.xml` (default), `amber19-all.xml` + `amber19/opc.xml` | loaded and asserted by a non-GPU test, including the Na+/Cl- templates |
| OS | ubuntu-latest (CI), Linux x86-64 (local) | no other OS is claimed |
| Accelerator (CI) | **CPU only** | the runners have no GPU; `environment-ci.yml` omits the CUDA pin |
| Accelerator (runs) | **CUDA by default** | generated scripts refuse a silent CPU fallback; locally verified on RTX A5000 + RTX 3080 |

No second Python or OpenMM version is listed, because none has been run. Adding one means adding it
to the workflow matrix and seeing it pass first.

## What portability means here

**Precise claims, in descending strength.**

1. **A generated project moves.** `sys-gen` writes `inputs/` and `md-gen` writes `MD/`, and `MD/`
   addresses `inputs/` by a relative path. Moving the two together to another machine needs no
   edit. The only absolute path written is the recorded interpreter in `run.sh`, which falls back
   to whatever `python3` provides.

2. **The generated scripts do not depend on this package.** They import OpenMM, PyYAML and the two
   modules copied in beside them. A project keeps working after the checkout is deleted; a test
   asserts no generated file names the checkout or imports `md_tools`.

3. **Binary checkpoints are environment-specific.** They give exact same-environment continuation
   and must never be described as portable. Moving a run directory between machines or OpenMM
   builds may make them unloadable.

4. **A serialized State is a portable fallback, but not a bitwise one.** It carries positions,
   velocities, box vectors, time and parameters, so continuation is physically valid and statistics
   are preserved. It does **not** restore a stochastic integrator's internal stream, so the
   trajectory diverges from what an uninterrupted run would have produced. The fallback is
   announced on use and recorded in the run summary.

5. **Rebuilding from the original structure may be scientifically consistent without being bitwise
   identical.** Parameterisation depends on the toolkit versions recorded in each build record
   in `machine.yaml` and by `sys-gen` in `inputs/provenance.yaml`. Reproducing a build exactly
   requires reproducing that environment; moving the already-built `inputs/` does not.

**Not claimed:** cross-machine bitwise reproducibility of dynamics, in any configuration.

## Scientific scope of the defaults

The evidence for every default, classified by strength, is in
[`scientific-defaults.md`](scientific-defaults.md). In summary:

| route | status | why |
|---|---|---|
| explicit, peptide or protein, ff14SB + TIP3P | supported | each pair is internally consistent; see §3 |
| explicit, protein + Sage ligand, ff14SB + TIP3P | supported | §3.3 — the specific triple with Sage 2.2.1 is an extrapolation from the Sage 2.x benchmarks, and is labelled one |
| explicit, ff19SB + OPC | supported alternative | §3.4 — no joint benchmark of ff19SB with Sage exists |
| implicit, peptide or protein, ff14SB + GBn2/mbondi3 | supported; Amber `igb=8` parity claimed | §5 |
| implicit, Sage small molecule | **experimental**, recorded as such in `forcefield.json` | §6 — mbondi3 reduces to mbondi2 for a one-residue ligand, and any element outside {H, C, N, O, S} gets GB-Neck2's unfitted fallback |
| 2 fs, unmodified hydrogen masses | supported baseline | §11.1 |
| 4 fs with HMR at 3.024 amu | supported for stability and equilibrium free energies; **not** for kinetics | §11.3 |
| AIS switching along the REST2 tau path | implemented and tested against a static REST2 rung to 0 kJ/mol and 0 kJ/mol/nm; **no free-energy estimator, forward only, fixed volume, no pV work** | `docs/journal/2026-08-27_ais-method-and-release-gaps.md` |

## Scientific status, which portability does not address

Mechanical portability is not scientific validity. No ladder is validated by any of this, the
2 fs / 4 fs hydrogen-mass-repartitioning equivalence gate is open, and every run in CI is
picoseconds long and proves execution only. Nothing in this repository has been validated against
experiment; every benchmark cited in the rationale was run by someone else, on their systems, with
their protocol.

Deletion is not evidence either. Removing the code that described the old architecture says nothing
about whether the current simulations are correct; that question is answered only by the tests and
the runs recorded in `docs/journal/`.
