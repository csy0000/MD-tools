# Supported versions and what portability means

## Matrix

Determined by what actually passes, not by aspiration. A version enters this table when a CI run
gates it, and is described as *locally verified* until then. The only workflow is
`release-validation`, which runs on `workflow_dispatch` and on an `openmm-v*` tag — not on every
push — so "gated" here means gated at release, not continuously.

| | version | status |
|---|---|---|
| Python | 3.12 | pinned by `environment-ci.yml`; gated by `release-validation`; locally verified on 3.12.13 |
| OpenMM | 8.6.0 | pinned by `environment-ci.yml`; gated by `release-validation`; locally verified on 8.6.0 |
| OpenFF toolkit | 0.19.0 locally | installed and import-checked by `md-template install`; unpinned in the solve |
| openmmforcefields | 0.16.0 locally | as above |
| AmberTools | `sqm`, `antechamber`, `tleap` on PATH | presence and AM1-BCC readiness checked at install and in CI |
| ParmEd / RDKit | 4.3.1 / 2026.03.1 locally | installed and import-checked |
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
   asserts no generated file names the checkout or imports `md_templates`.

3. **Binary checkpoints are environment-specific.** They give exact same-environment continuation
   and must never be described as portable. Moving a run directory between machines or OpenMM
   builds may make them unloadable.

4. **A serialized State is a portable fallback, but not a bitwise one.** It carries positions,
   velocities, box vectors, time and parameters, so continuation is physically valid and statistics
   are preserved. It does **not** restore a stochastic integrator's internal stream, so the
   trajectory diverges from what an uninterrupted run would have produced. The fallback is
   announced on use and recorded in the run summary.

5. **Rebuilding from the original structure may be scientifically consistent without being bitwise
   identical.** Parameterisation depends on the toolkit versions recorded by `md-template install`
   in `machine.yaml` and by `sys-gen` in `inputs/provenance.yaml`. Reproducing a build exactly
   requires reproducing that environment; moving the already-built `inputs/` does not.

**Not claimed:** cross-machine bitwise reproducibility of dynamics, in any configuration.

## Scientific status, which portability does not address

Mechanical portability is not scientific validity. No ladder is validated by any of this, the
2 fs / 4 fs hydrogen-mass-repartitioning equivalence gate is open, and every run in CI is
picoseconds long and proves execution only.

Deletion is not evidence either. Removing the code that described the old architecture says nothing
about whether the current simulations are correct; that question is answered only by the tests and
the runs recorded in `docs/journal/`.
