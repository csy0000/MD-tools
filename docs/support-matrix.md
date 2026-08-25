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

`pydantic` is no longer a dependency of this package and has been removed from the table.

No second Python or OpenMM version is listed, because none has been run. Adding one means adding it
to the workflow matrix and seeing it pass first.

## What portability means here

> **Partly historical.** Points 1, 2 and 5 describe the removed bundle architecture. Points 3 and 4,
> on checkpoints and serialized States, still hold.

**Precise claims, in descending strength.**

1. **Transferred prepared artifacts are byte-identical when the checksums match.** `checksums.json`
   hashes the bytes of every required artifact and original input. A bundle that validates after
   relocation contains exactly the files it was prepared with.

2. **A relocated bundle runs.** Both canonical routes are prepared, copied to an unrelated
   directory, their originating directories deleted, and then validated, inspected, run and
   resumed from the copies — with the package installed from a wheel and no checkout on the path.

3. **Binary checkpoints are environment-specific.** They give exact same-environment continuation
   and must never be described as portable. Moving a run directory between machines or OpenMM
   builds may make them unloadable.

4. **A serialized State is a portable fallback, but not a bitwise one.** It carries positions,
   velocities, box vectors, time and parameters, so continuation is physically valid and statistics
   are preserved. It does **not** restore a stochastic integrator's internal stream, so the
   trajectory diverges from what an uninterrupted run would have produced. The fallback is
   announced on use and recorded in the run summary.

5. **Rebuilding from original inputs may be scientifically consistent without being bitwise
   identical.** Parameterisation depends on the toolkit versions recorded in `environment.json`.
   Reproducing a bundle exactly requires reproducing that environment; transferring the prepared
   bundle does not.

**Not claimed:** cross-machine bitwise reproducibility of dynamics, in any configuration.

## Bundle schema versions

> **Historical.** The bundle architecture described below was removed in `ca29fcd`. `sys-gen` now
> writes a plain `inputs/` folder and `md-gen` a plain `MD/` project. This section is retained to
> interpret validation reports produced before that change, and describes nothing in the current
> package.

| version | guarantees |
|---|---|
| 2 | checksums, original inputs, force-field and environment provenance, separated topology-atom / OpenMM-particle / virtual-site / massless counts, mmCIF topology, canonical configuration and hashes |
| 1 | readable and runnable through the compatibility path; **none** of the above. `bundle validate` reports it as `v1-compatibility` and names what is missing |

A version-1 bundle is never reported as satisfying the version-2 contract.

## Scientific status, which portability does not address

Mechanical portability is not scientific validity. No ladder is validated by any of this, the
2 fs / 4 fs hydrogen-mass-repartitioning equivalence gate is open, and every run in CI is
picoseconds long and proves execution only.
