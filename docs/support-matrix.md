# Supported versions and what portability means

## Matrix

Determined by what actually passes, not by aspiration. A version enters this table when a CI run
gates it, and is described as *locally verified* until then.

| | version | status |
|---|---|---|
| Python | 3.11 | gated by `fast` and `integration-cpu`; locally verified on 3.11.15 |
| OpenMM | 8.5.1 | locally verified; pinned by `environment-ci.yml` |
| pydantic | ≥ 2 (2.11.10 locally) | gated by the unit suite |
| OS | ubuntu-latest (CI), Linux x86-64 (local) | no other OS is claimed |
| Accelerator | **CPU only** | CUDA is neither required nor tested here |

No second Python or OpenMM version is listed, because none has been run. Adding one means adding it
to the workflow matrix and seeing it pass first.

## What is implemented, and how far it is tested

Four levels, kept apart because they are different claims. The campaign's whole documentation
discipline is that nothing moves up a row without evidence.

| capability | level |
|---|---|
| catalog listing, inspection, identity (`md-templates list/inspect/identity`) | **implemented + CI-tested**, engine-free, gated by the fast checks |
| packaged catalog, resource integrity, build provenance | **implemented + CI-tested** in the strict wheel gate |
| conventional MD: prepare, run, resume — generic and legacy routes | **implemented + CI-tested** on CPU, both routes proven to produce identical canonical hashes |
| REST2: prepare, run, resume — generic and legacy routes | **implemented + CI-tested** on CPU, three replicas committed and resumed |
| bundle relocation, crash recovery, State fallback | **implemented + CI-tested** (slow gate, real SIGKILL) |
| CUDA / OpenCL / `--device` / `Precision` | **implemented, contract-tested, real GPU not run here** — see below |
| cyclo-RGDfV ten-rung REST2 | **system-specific pilot-supported evidence**, not general validation |
| macrocycle eight-rung example | **unvalidated example** |
| general conventional MD, general REST2 | **scientifically unvalidated** |

CPU smoke runs are engineering evidence that the pipeline executes. They are not scientific
validation, and `cpu-smoke-v1` is refused as scientific evidence by the descriptor model itself.

## GPU and platform contract

CI is CPU-only, but GPU support is an operational contract and is preserved deliberately:

* `CUDA` and `OpenCL` remain selectable, with explicit `--device` validation;
* OpenMM `DeviceIndex` and `Precision` (`single` / `mixed` / `double`) are passed through unchanged,
  and the effective default is unchanged;
* an unavailable platform or device fails explicitly — there is **no silent fallback to CPU**;
* REST2 remains one process with one selected device. No implicit multi-GPU replica distribution was
  added; external one-process-per-GPU launching remains the approach.

**Status, stated honestly: implemented and contract-tested; real-GPU runs were not performed for
this work.** No CUDA runner was available. Mocked selection tests and CPU runs are not CUDA
validation, and nothing here should be read as such.

## What portability means here

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

| version | guarantees |
|---|---|
| 2 | checksums, original inputs, force-field and environment provenance, separated topology-atom / OpenMM-particle / virtual-site / massless counts, mmCIF topology, canonical configuration and hashes |
| 1 | readable and runnable through the compatibility path; **none** of the above. `bundle validate` reports it as `v1-compatibility` and names what is missing |

A version-1 bundle is never reported as satisfying the version-2 contract.

## Scientific status, which portability does not address

Mechanical portability is not scientific validity. No ladder is validated by any of this, the
2 fs / 4 fs hydrogen-mass-repartitioning equivalence gate is open, and every run in CI is
picoseconds long and proves execution only.
