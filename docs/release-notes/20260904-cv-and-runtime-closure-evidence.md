# Runtime closure and torsion collective variables: what was corrected, and the evidence

Baseline `0498c22`; instruction `597cf97`. Every number here was measured on this
machine, at a named commit, with real CUDA devices, a real MPI launcher and a real
installed wheel. Where a claim has no measurement behind it, it is not made.

## Implementation commits

| commit | what |
|---|---|
| `a9dad76` | cMD completion is proven by recomputing the fingerprint and verifying every digest; `--overwrite` is one transaction over the complete owned inventory |
| `34084ca` | AIS directory identity and resume semantics: one decision, verified ownership, no destructive restart |
| `1002c47` | the strict `cv.yaml` schema and the torsion arithmetic |
| `298c667` | a rank-local failure stops the whole communicator instead of hanging it |
| `5431401` | the rREST2 reservoir is validated in the preflight, before `-odir` exists |
| `1c744ba` | one ladder plan, built before output and consumed by the driver |
| `01c99e0` | the CV configuration section, the exact schedule, and the series writer |
| `e57a0de` | cMD reports torsions on their own cadence |
| `e6b4dfa` | the definition is resolved against the config and copied in content-addressed |
| `006081d` | REST2/rREST2 report per thermodynamic state |
| `7c97406` | AIS reports per path, aligned to the coordinates actually saved |
| `e05cada` | the documentation, across every surface |
| `840c97e` | the new CV sites registered in the CUDA inventory |

## Test lanes

| lane | command | result | wall |
|---|---|---|---:|
| fast | `pytest tests -m "not slow and not gpu"` | **1252 passed**, 0 failed, 251 deselected | 207.60 s |
| CUDA + slow | `pytest tests -m "gpu or slow"` | **250 passed**, 0 failed, 1 skipped, 1252 deselected | 1343.29 s |
| MPI | `pytest tests/test_mpi_fail_closed.py tests/test_driver_fail_closed.py` | **29 passed**, 0 failed | 115.14 s |
| CUDA evidence | `pytest tests/test_cuda_coverage_matrix.py --cuda-evidence=…` | **27 passed** | 483.60 s |
| wheel | `python -m build --wheel`, `pip install --no-deps` into a fresh venv | built, installed, origin-verified | — |

The one skip is `test_write_the_coverage_evidence`, which writes the matrix only when
`--cuda-evidence` is given so an ordinary GPU run does not rewrite a committed file as a
side effect. It is not skipped in the evidence lane above, where it ran and wrote.

Nothing was xfailed or deselected by hand.

### What the lanes cost to get green

The first full `gpu or slow` run was **212 passed, 14 failed, 24 errors**. Every failure
reduced to one of three causes, and all three are recorded here rather than smoothed over:

* **`md-openmm` was not on `PATH`** in the shell the lane ran in, so every test that shells
  out to the installed console script failed with `command not found`. An environment gap,
  not a defect. The lane was re-run with the wheel venv's `bin` on `PATH` — which is
  stronger evidence anyway, because it exercises the *installed* entry point rather than a
  source checkout.
* **The CUDA coverage matrix refused three new functions.** That is what it is for: the AST
  check fails on a Context-touching function nobody classified, rather than letting it go
  untested. Resolved in `840c97e` by classifying each honestly (below).
* **A test of ours leaked a netCDF handle.** The empty-reservoir test opened a
  `netCDF4.Dataset` to assert a frame count and never closed it, so the handle held the file
  and the next test in that module could not rewrite the path. It failed only when the whole
  lane ran in one process, and was green when the file ran alone.

## Hardware, MPI and versions

* 9 CUDA devices: 1 × NVIDIA RTX A5000 (24564 MiB), 8 × NVIDIA GeForce RTX 3080
  (10240 MiB each); driver 580.173.02
* Open MPI 5.0.8 (launched as `prterun`), mpi4py 4.1.2, at 2 ranks in the failure-injection
  lane
* Python 3.12.14, OpenMM 8.6.0.dev-c6173db
* Linux 6.8.0-124-generic, x86-64

## The corrected lifecycle refusals, each with its direct evidence

| refusal | evidence |
|---|---|
| a completed cMD stage whose fingerprint, digests, particle count, checkpoint or stream counts disagree | `test_cmd_completion_verification.py` — 13 tests, each damaging one output in a way that leaves it *existing* |
| `--overwrite` leaving any owned output behind, or a crash mid-replacement | `test_cmd_overwrite_transaction.py` |
| an AIS directory with owned artefacts and no readable identity | `test_ais_directory_identity.py` — refused under *every* flag combination, including `--overwrite` |
| an interrupted AIS path restarted destructively because `--resume` was omitted | `test_ais_directory_identity.py`, end to end: the path is interrupted after its first checkpoint commits, the re-run without `--resume` is refused, and the committed checkpoint is proven byte-identical before and after |
| a missing or empty rREST2 reservoir | `test_reservoir_preflight.py` — refused with `-odir` **not created** afterwards |
| a reservoir recorded under a different Hamiltonian than the top rung | `reservoir.validate_source_read_only`, called from the preflight |
| a ladder plan whose rung count disagrees with the protocol | `test_ladder_plan.py` |
| a CV interval that does not divide its enclosing span | `test_cv_reporting.py`, `test_remd_cv_output.py`, `test_ais_cv_output.py` |

## MPI: fail-closed, not hung

`test_mpi_fail_closed.py` runs a real `mpirun -n 2` REST2 ladder and injects a rank-local
failure *inside the dynamics loop*, parametrised over which rank fails — root and non-root.
The subprocess timeout **is** the assertion: a hang is the failure being tested for, so a
test that waits forever cannot detect one. Both cases stop the whole communicator and
neither leaves a completion manifest. Both passed.

`test_driver_fail_closed.py` covers, in-process and without a launcher, the case the
launcher tests cannot easily provoke: a **second** failure while reporting the first. A full
disk or a stalled filesystem is often *why* the first failure happened, and both surface
again in the handler. Unwrapped, that secondary exception skipped the abort entirely and
turned a stopped job into a hung one. Each of these tests was confirmed to fail against the
unhardened code, not merely to pass against the new.

## Source-to-CUDA coverage

Counted against what the AST matcher actually found, at commit `840c97e`, not `-dirty`:

* **29** source functions perform a CUDA-relevant operation
* **25** carry a lane
* **4** are recorded as not reaching a device
* **0 unclassified**

The three sites added this round, classified honestly:

* `ais/run.py::run_one_path.write_cv` — **a CUDA site.** It calls
  `context.getState(getPositions=True)`, which on CUDA is a device synchronise and a
  device-to-host copy. Positions only, never `getEnergy`, so a CV observation costs no
  Hamiltonian evaluation and cannot be confused with one in the cost accounting.
* `md/cv_report.py::CVReporter.observe` — **not a CUDA site.** It reads positions off a
  `State` it is handed and evaluates torsions on the host in numpy. It constructs nothing and
  asks the Context for nothing.
* `md/completion.py::verify_completed_stage` — **not a CUDA site.** It deserialises a `State`
  from XML *on disk*; the same `getPositions` spelling as a live Context, but the object is a
  file's contents, and the verifier is entirely read-only.

## The installed wheel

```text
wheel  : md_tools-0.5.0.dev0-py3-none-any.whl
origin : …/wheelbuild/venv/lib/python3.12/site-packages/md_tools/__init__.py
```

The origin check is an assertion, not a print: it fails if `md_tools` resolves into the
source checkout. Against that install, outside the checkout:

* all four commands answer — `build-top`, `build-md`, `md-run`, `data-register`
* `build-top` built an implicit ALA system (GBn2)
* `build-md` generated a cMD project **with `collective_variables` enabled**
* the run completed and produced a CV series with the schedule the configuration asked for:

```text
step,time_ps,trajectory_frame_index,phi
0,0.000000,,-175.645071
5,0.010000,,-175.478503
10,0.020000,0,-173.655555
15,0.030000,,-169.616825
20,0.040000,1,-165.287315
```

Step 0 and the final step appear exactly once; the CV cadence (5) is finer than the
trajectory cadence (10), so rows without a stored frame carry an **empty**
`trajectory_frame_index` rather than `0` or `-1`, both of which are real frame indices.

## Collective variables: what is actually verified

* **Numerical.** Every reported value is compared against an independent calculation, never
  merely checked for column existence. An analytically constructed four-atom geometry fixes
  the expected angle by construction; MDTraj is the independent implementation for real ALA
  torsions. The comparison is by wrapped angular difference, because the two implementations
  wrap to different half-open intervals — the `[-180, 180)` boundary is asserted separately.
* **Sign convention.** The first implementation was inverted by exactly 180°, which produces
  a full column of plausible in-range numbers, every one wrong. Caught by the analytic
  construction and confirmed against MDTraj. Two test-design faults surfaced with it and are
  documented in the tests: the shipped ALA fixture is fully extended, so both backbone
  torsions sit at exactly 180° where a sign flip passes — the coordinates are now perturbed
  first, with an assertion that they moved off the degenerate value; and a 1 nm bond in a
  2 nm box sits exactly on a minimum-image tie where either answer is defensible.
* **Periodicity.** The minimum image is brute-forced against a 9×9×9 neighbourhood and the
  result additionally proven to be a genuine lattice translate of the input, for both
  orthorhombic and triclinic cells.
* **Hamiltonian invariance.** Asserted in the half that can be exact — identical System
  serialisation, force inventory, force groups and single-point energy with reporting on and
  off — and, for coordinates after dynamics, against a **measured** reproducibility floor.
  Two runs of the same configuration are not bit-identical on the CPU platform (~9 × 10⁻⁹ nm
  on this system), so an exact-equality assertion there would have been testing platform
  determinism rather than anything about collective variables. An earlier version of that
  test did exactly that and failed for a reason unrelated to the feature.
* **Alignment.** For AIS, each row carrying a `coordinate_frame_index` is recomputed from the
  frame stored in that path's own NetCDF. A one-frame misalignment fails it.
* **State versus walker.** Reading `walker_index` across the state files at any fixed step
  must give a permutation of the walkers — each walker occupies exactly one state. A
  companion test checks the walkers actually moved and **skips with an explanation** rather
  than passing silently if the short run happened to accept no exchange, which would make the
  permutation test vacuous.
* **Portability.** The generated tree is copied somewhere the original `cv.yaml` does not
  exist and run there; it must still produce its series.

## Limitations, stated plainly

* **v1 is torsions only.** No distances, angles, RMSD, coordination numbers or combinations,
  and no biasing of any kind. An unsupported `type` is refused by name with the schema
  version, so a definition written against a future schema fails loudly rather than
  silently producing a shorter CSV.
* **The short runs here are mechanism tests.** Twenty to forty steps, two to four exchanges,
  two AIS paths. They verify that the bookkeeping is correct; they support **no** claim about
  convergence, sampling quality or any physical result.
* **CPU-platform runs are not bit-reproducible.** Measured above. Any comparison of
  coordinates after dynamics on that platform is a tolerance comparison, and the tolerance
  used here is a measured floor rather than a chosen constant.
* **The MPI failure-injection lane runs at 2 ranks.** Larger worlds are exercised by the
  other GPU lanes but not with injected rank-local failures.
* **`AIS_cv.csv` is assembled only from paths whose completion manifest verified.** A
  campaign with failed paths produces an aggregate describing fewer paths than were
  requested, deliberately, and the per-path files remain for inspection.
