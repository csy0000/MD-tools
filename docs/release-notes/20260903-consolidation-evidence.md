# Runtime consolidation: what was corrected, and the evidence

Baseline `5571982`; instruction `e0b8bd1`. Every number here was measured on
this machine, at a named commit, with a real CUDA device and a real MPI
launcher. Where a claim has no measurement behind it, it is not made.

## Implementation commits

| commit | what |
|---|---|
| `422d7a6` | AIS decides its run identity before any output exists; the cost counters stop calling committed work discarded |
| `9ce01ad` | AIS finalization is crash-atomic across ten boundaries |
| `0654618` | the cMD plan carries the prepared System; the whole stage chain is planned before stage 1 |
| `eaefa5c` | every post-preflight rank-local failure is collective |
| `590c507` | one ladder route, one plan, one overwrite policy |
| `4028ceb` | the CUDA inventory covers reporters and CUDA-derived output |
| `44355a4` | the CUDA lanes' counter assertions, which Slice A had left stale |
| `a70bab9` | the ladder fixture copies inputs, not a directory that has been run in |
| `b4ba146` | the coverage counts are counted against what was found |
| `d9b581f` | the evidence artefacts for `b4ba146` |

## Test lanes

| lane | command | result | wall |
|---|---|---|---:|
| fast | `python -m pytest tests -m "not slow and not gpu"` | **1143 passed**, 0 failed, **0 skipped**, 208 deselected | 431.65 s |
| CUDA + MPI | `python -m pytest tests -m "gpu or slow"` | **208 passed**, 0 failed, **0 skipped**, 1144 deselected | 2355.54 s |
| wheel | `python -m build --wheel`, `pip install --no-deps` into a fresh venv | built and installed | — |

Nothing was skipped, xfailed or deselected by hand in either lane. The CUDA
lane's own record is [`cuda-coverage-matrix.md`](cuda-coverage-matrix.md) and
[`cuda-coverage-matrix.json`](cuda-coverage-matrix.json), stamped
`b4ba146` and not `-dirty`.

## Hardware, MPI and versions

* 9 CUDA devices: 1 × NVIDIA RTX A5000 (24564 MiB), 8 × NVIDIA GeForce RTX 3080
  (10240 MiB each); driver 580.173.02
* Open MPI, launched as `prterun`, at 2, 4 and 6 ranks
* Python 3.12.14, OpenMM 8.6.0.dev-c6173db (conda-forge release `py312hdfcc665_0`)
* Linux 6.8.0-124-generic, x86-64

Rank-to-device placement, from a 2-rank ladder run through the INSTALLED wheel:

```text
# platform : CUDA device=0 (machine.openmm.device_policy: local_rank (rank 0 of 2)), precision mixed
# platform : CUDA device=1 (machine.openmm.device_policy: local_rank (rank 1 of 2)), precision mixed
```

## The installed wheel

```text
import origin : .../wheel2/lib/python3.12/site-packages/md_tools/__init__.py
```

Run from `/tmp`, outside the checkout, against that install:

* all four commands answer: `build-top`, `build-md`, `md-run`, `data-register`
* `build-top` built an implicit ALA system (GBn2)
* `build-md` generated a cMD project and a 2-state REST2 project
* the whole cMD chain — `min`, `eq_nvt_posres`, `eq_nvt_posres_2`,
  `eq_nvt_free`, `cMD` — ran to `run_status: completed` with
  `platform CUDA`
* `mpirun -n 2 md-openmm md-run -ng 2` ran the REST2 ladder to
  `run_status: completed`, writing `remd0.nc`, `remd1.nc`, `rem.log`,
  `REST2.out` and `REST2.out.rank01`

Two invented configuration keys were REFUSED by the schema with a suggestion
rather than ignored (`dynamics.production_steps`, `rest2.exchange_interval_ps`).
That is the intended behaviour and it is recorded here because it was met.

## Source-to-CUDA coverage

* 26 source functions perform a CUDA-relevant operation, by the operation
  matcher: construct, integrate, minimize, evaluate, parameters, checkpoint,
  **report** and **derive** — the last two added this round
* of those, **24 carry a lane** and **2 are recorded as not reaching a device**
  (`solvate` reads box vectors off a Topology; `_read_initial_configuration`
  reads a State deserialised from XML)
* **0 unclassified.** `test_every_cuda_operation_in_the_source_is_in_this_matrix`
  fails the suite if that stops being true, and a companion test plants an
  unclassified reporter to prove the check can fail
* 2 further functions carry a lane without being flagged by the matcher
  (they resolve a Platform rather than touch a Context)
* 23 lanes ran

## AIS reconstruction and frame-aligned recomputation

Measured on CUDA, mixed precision:

| check | worst error |
|---|---|
| three-group sum against the direct total, explicit solvent (PME) | 1.082e-04 kJ/mol over 6 rows |
| HS rows recomputed in a fresh Context at their own saved frames, implicit | 8.275e-05 kJ/mol over 3 rows |
| HS rows recomputed in a fresh Context at their own saved frames, explicit | 4.488e-01 kJ/mol over 3 rows |

The explicit-solvent recomputation tolerance is four orders of magnitude
looser than the implicit one. That is the reciprocal sum in mixed precision,
not a defect, and it is stated rather than averaged away.

## Cost accounting

The counters are non-overlapping and their relationships are published in the
record itself:

```text
useful_total = direct_work + work_basis_probe + observation_potential + other_useful
paid_total   = useful_total + known_discarded
```

The two probe counters are separate because they are taken at DIFFERENT
COORDINATES: the work-basis probe at the frozen pre-switch `x_j`, the
observation probe at the saved coordinate `x_t`. Measured on CUDA:
64 evaluations (24 work-basis probe, 16 work, 24 observation) with 64
parameter updates in 0.498 s for the decomposition lane; 162 evaluations
(90 / 60 / 12) for the HS recomputation lanes.

**What cannot be observed.** A committed checkpoint cannot know how much work
happened after it and before the crash. That cost is real and unrecoverable,
and the record says so with `discarded_is_complete: false` after a resume
rather than reporting a number it cannot have. An uninterrupted run asserts
`known_discarded == 0` and `discarded_is_complete == true`.

## Remaining limitations

* The explicit-solvent HS recomputation tolerance above is 4.5e-01 kJ/mol.
* Every lane here is picoseconds long. They prove mechanism, not sampling, and
  no result in this document supports any convergence or free-energy claim.
* CI runs on CPU only: the GitHub runners have no GPU, so the CUDA and MPI
  evidence is local and is not reproduced by the workflow.
* The counts and tolerances are from ONE machine. Nothing here claims
  cross-machine bitwise reproducibility of dynamics.
