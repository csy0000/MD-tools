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
| CUDA / `--device` / `Precision` | **implemented, contract-tested, and run on real hardware** — see below |
| OpenCL | **implemented and contract-tested**; not exercised on real hardware |
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

### Real-hardware run, 2026-08-19

`scripts/ci/gpu_contract_run.sh` was run on a machine with nine GPUs — one RTX A5000 and eight
RTX 3080s — and passed end to end.

| | |
|---|---|
| device selected | index 1 → **NVIDIA GeForce RTX 3080**, PCI `00000000:1B:00.0` |
| driver | 580.173.02 |
| platform / precision | CUDA / `mixed` |
| conventional MD | prepared through **both** routes, run and resumed: generation 3, 2 invocations |
| REST2 | prepared, run and resumed on one device in one process: replicas 00, 01, 02 committed |
| unavailable device | `--device 99` refused, no downgrade to CPU |

Two results worth stating precisely.

**Device indices are stable because `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set at launch.** On this
machine that is not academic: with an A5000 among eight 3080s, CUDA's default ordering is by
capability, so an index would not match `nvidia-smi`. `validate-env` confirms the variable is set and
reports the device it resolved to.

**The scientific hashes are platform-independent, and this was measured rather than assumed.**
Preparing the same document on CPU and on the RTX 3080 gives identical `system_build`,
`prepared_state`, `protocol` and `protocol_at_prepare` hashes; only `execution` differs
(`6f0a2135…` on CPU, `f4460c86…` on CUDA device 1), because that projection exists to record platform,
device and precision and is deliberately outside every compatibility decision. A bundle prepared on
CPU and one prepared on a GPU are interchangeable.

**What this is not.** Picoseconds of dynamics that completed. It is engineering evidence that the
CUDA path works, exactly as the CPU gate is engineering evidence for that path. It validates no
science, no ladder and no force field. OpenCL remains contract-tested only — the code path is shared
with CUDA but no OpenCL run was performed.

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
