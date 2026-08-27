# 2026-08-27 — AIS, and the two release gaps that came before it

Instruction: `claudecode-instructions/20260827_ais-method-and-release-gap-correction.md`.
Branch: `dev`. Version: `0.4.0.dev0` (unchanged). Executed in the instruction's order.

---

## 1. The explicit force-field/water pairing gap

`sys-config` writes a coupled selection — ff14SB with TIP3P, ff19SB with OPC — but the file it
writes is ordinary editable YAML. Generating the pair correctly is not the same as building it
correctly: changing `solvent.model` to `OPC` and leaving `forcefield.protein` at ff14SB produced a
System, ran to completion, and reported a Hamiltonian nobody validated.

`config._check_explicit_pairing` now refuses that, in both directions and on both halves, before a
System exists. It matches on the force-field **family** rather than an exact string, because there
is more than one way to spell each resource (`amber14-all.xml`, `amber14/protein.ff14SB.xml`,
`ff14SB.xml`), and the crossing is the thing being caught, not the spelling. The message names the
mismatched dotted field, its value, what it should be, and both supported pairs.

A resource belonging to **neither** family is left alone. Somebody loading a force field this
repository does not ship is doing something deliberate, and this table has no opinion about it; the
check exists to catch a crossing between the two pairs it does know.

The ligand-only route is unaffected in the record: validation asks the FILE to name a complete
selection so it stays unambiguous, while `forcefield.json` still reports `protein.openmm_resource:
null` for a build that loaded no protein XML. Both facts hold at once, and a test asserts each.

Tests: 6 crossed pairs refused, 3 generated pairs still resolve, an unknown force field passes
through, and the ligand record still names no protein. None builds a System.

---

## 2. Exact OpenMM 8.6.0 — and a correction to what "exact" can be decided from

### The premise turned out to be wrong, in an instructive way

The instruction asked to distinguish the stable 8.6.0 release from "a development build whose
version string merely begins with 8.6.0", because the previous acceptance run reported
`8.6.0.dev-c6173db`.

It is not a development build. The conda-forge **release** package reports that string:

```
conda package        : openmm 8.6.0 py312hdfcc665_0 from conda-forge
openmm.version.version       : 8.6.0.dev-c6173db
openmm.version.short_version : 8.6.0
openmm.version.git_revision  : c6173db6e8edd705eb59172bd21e9ce69c572405
```

Upstream stamps the build commit into the version string, and conda-forge builds the release from
that commit. `conda search -c conda-forge openmm` shows **only** `8.6.0` in the 8.6 series — there
is no `8.6.0.dev` or `8.6.0rc` package. And the pre-existing working environment
(`openfftools860`) carries `openmm 8.6.0 py312h5a97af1_0`, also from conda-forge.

**So the 0.4 defaults acceptance run was already on the exact stable release.** Only its version
string looked like a development build. That is recorded here rather than quietly fixed, because
the earlier report stated the version string as if it identified the build.

### What the check is actually built on

Neither Python-level string can decide the question:

* `openmm.version.version` carries a build marker for a release, so `.dev` does not mean
  development;
* `openmm.version.short_version` is `8.6.0` for the release and for any release candidate of it.

The authoritative fact is the **installed package identity** in `conda-meta/openmm-*.json`, where a
prerelease is a different package version (`8.6.0rc1`, not `8.6.0`). `install.openmm.release_status`
judges on that, falls back to the Python string only when there is no package metadata (a pip or
source install), records both, and records their disagreement as a fact rather than hiding it.

`md-template install --validate PREFIX --expect-version 8.6.0` applies the rule. Without
`--expect-version` nothing is refused: a user validating an environment they built around a
different OpenMM gets the facts and no verdict, which is what `--validate` is for.

### Two real defects found while establishing the environment

**The CUDA solve was unbounded above.** `md-template install` asked for `cuda-version>=11.8` and
let the solver take the newest. On this machine (driver 580.173.02, ceiling **CUDA 13.0**) it chose
**CUDA 13.3**. The environment installed cleanly, passed validation, and then failed every real
System:

```
openmm.OpenMMException: Error loading CUDA module: CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)
```

The solve is now capped at the ceiling `nvidia-smi` reports (`driver_cuda_ceiling()`), and the
rebuilt environment carries `cuda-version 13.0` and runs.

**The environment probe was too weak to catch it.** The old `one_step` check built a System of two
free particles, which creates a context on a mismatched CUDA build without complaint — measured
directly:

```
two free particles : ok
with real forces   : OpenMMException: Error loading CUDA module: CUDA_ERROR_UNSUPPORTED_PTX_VERSION
```

The probe now carries a `NonbondedForce` and a `HarmonicBondForce` and integrates with
`LangevinMiddleIntegrator`, so it compiles the kernels a real run compiles. Both are tested.

`mdtraj` was added to `CONDA_PACKAGES`, `REQUIRED_IMPORTS`, the probe and `environment-ci.yml`,
because the advertised AIS command reads its source trajectory with it.

### The acceptance environment

```
$MD_STACK/envs/openmm-8.6.0
  mamba create --yes --prefix ... -c conda-forge openmm=8.6.0 python=3.12 pyyaml numpy
      openff-toolkit openff-nagl-models openmmforcefields ambertools parmed rdkit mdtraj
      cuda-version>=11.8 cuda-version<=13.0
  -> openmm 8.6.0 py312hdfcc665_0 (conda-forge)  ->  release check: exact stable release
  -> mdtraj 1.11.1, openff-toolkit 0.19.0, openmmforcefields 0.16.0, parmed 4.3.1,
     rdkit 2026.03.1, python 3.12.14, cuda-version 13.0
```

---

## 3. AIS

### What was taken from the earlier implementation, and what was not

`csy0000/partitioned-REST2` (`partition-consensus-test`, `src/escort_ais/methods`) was read for
three ideas: source-frame selection, exact REST2 switching, and nonequilibrium work bookkeeping.
Nothing was copied. Deliberately **not** brought across: the approximate `energy_interpolation`
route, umbrella/partitioning logic, BAIS-specific logic, consensus logic, and the on-device
optimisation framework.

### One source of truth for the scaling

The rules live in `templates/rest2_scaling.py`, which every generated project already carries.
`TauSwitcher` was added there rather than in a new module:

* it holds a private, never-modified clone of the base System;
* `set_tau` restores the **unscaled** parameters from that clone and then calls the same
  `_scale_nonbonded` / `_scale_torsions` / `_scale_cmap` functions a static rung is built with,
  before `updateParametersInContext`;
* `CustomGBForce` is handled through the global scale parameter, which is why
  `build_scaled_system` grew a `prepare_for_switching` flag: at tau = 0 the scaling is a no-op and
  a REST2 rung wants the untouched System, but an AIS path starting there needs the parameter
  already compiled into the expressions.

Restoring from the base rather than composing on the previous tau is what makes switching away and
back land on the identical Hamiltonian instead of a compounded one. A test checks exactly that.

Rebuilding the System per update was the obvious alternative and is not viable: the default
schedule updates every step, and serialising a solvated System tens of thousands of times would
dominate the run.

### The schedule is computed once

`openmm/ais.py` derives the path — which taus are visited, when the parameters change, which points
are observed — and `md-gen` writes the result into `AIS/path_definition.yaml`. The generated runtime
reads it and never recomputes it. Deriving it at both ends is how a project ends up observing a
schedule its own record does not describe.

Three divisibilities must hold exactly, and each is refused rather than rounded, naming the multiple
that would fit:

```
total_steps             = switching_duration_ps at this timestep
number_of_updates       = total_steps / parameter_update_interval_steps
updates_per_observation = number_of_updates / (number_of_observations - 1)
```

### The source contract

AIS does not start from the common chain, so **it is not in `run_all.sh`**. Its source is an
equilibrium trajectory at `tau_start`, and the two things a naive implementation would guess are
both refused:

* **the source tau.** Read from the trajectory's companion `resolved_run.yaml` — `tau` for a cMD
  run, `replicas[].tau` for a REST2 rung, matched by directory name. If it cannot be established
  the run stops. An ensemble equilibrated at a different tau makes the first work value absorb the
  mismatch, silently.
* **frame times.** A frame index is never treated as a time and a DCD header is never treated as
  the production clock. `trajectory_record` in `md_stages.py` now writes a `frame_time_map` —
  reporter interval, timestep, and the resulting first-frame time — at the point those are known,
  and the AIS runtime reads it. Failing that, `first_frame_time_ps` **and** `frame_interval_ps` are
  required together; one without the other is refused, because accepting one would leave the other
  to be guessed.

`start_time_ps` and `end_time_ps` are inclusive. Selection is uniform over eligible frames, without
replacement by default, seeded from the recorded master seed, and written to
`selected_initial_frames.csv` before anything propagates. Requesting more paths than there are
eligible frames is refused: two paths from one configuration are not two realisations.

Atom count and atom order are validated against `inputs/topology.pdb` — the System's parameters are
per index, so a reordered trajectory would be scaled atom-by-atom wrongly.

### One fix the first run forced

MDTraj rebuilds box vectors from the lengths and angles a DCD stores, in the usual lower-triangular
convention. OpenMM rejected them:

```
openmm.OpenMMException: Periodic box vectors must be in reduced form.
```

`reduced_box_vectors` subtracts integer multiples of the earlier vectors from the later ones. That
renames the lattice vectors without changing the lattice, and the runtime verifies the cell volume
is unchanged to 1e-9 before propagating — refusing rather than running in a box that is not the one
the source frame had. It matters here because the box this repository builds is a rhombic
dodecahedron.

### Fixed volume, and what that costs

No barostat is added, and the prepared System is checked to carry none. An explicit path keeps the
box of the frame it started from. **Pressure-volume work is not included**, even when the source
ensemble was NPT. That is stated in `path_definition.yaml`, in both runtime records, in the
generated script's own log line, in the README and in the example.

---

## 4. Acceptance

All of the following ran in the exact-release environment above.

### Tests

```
$ENV/bin/python -m pytest tests/ -q -m "not gpu and not slow"     140 passed, 95 deselected, 14 s
CUDA_VISIBLE_DEVICES=0,1 $ENV/bin/python -m pytest tests/test_ais.py -q     39 passed, 14 s
CUDA_VISIBLE_DEVICES=0,1,2,3 $ENV/bin/python -m pytest tests/ -q            254 passed, 152.89 s
```

254 passed, 0 failed, 0 skipped. The suite was 196 before this task; the 58 new tests are
`tests/test_ais.py` (39) plus additions to `test_config_generation.py` and `test_install.py`.

### Dynamic switching against a static REST2 rung — CUDA, double precision

```
  tau     static U (kJ/mol)             dynamic U       |dU|   max|dF| (kJ/mol/nm)
  0.0        1629.079048949        1629.079048949  0.000e+00             0.000e+00
 0.25        1670.141142294        1670.141142294  0.000e+00             0.000e+00
  0.5        1702.764876472        1702.764876472  0.000e+00             0.000e+00
```

Exact on the energy **and on every atom's force**. Forces are the stronger check: an energy can
agree by cancellation, a full force array cannot. Switching away to 0.4 and back reproduces the
same values, which is what proves the restore-from-base design.

Omega exclusion is checked on the switched torsion parameters directly: every solute torsion about
an omega bond keeps its force constant, every other solute torsion is scaled by exactly `s`, and no
environment torsion is touched.

### Work bookkeeping

Frozen coordinates, 40 parameter changes from tau 0.5 to 0:

```
frozen-coordinate work : -73.685827523 kJ/mol
U(tau_end) - U(tau_start): -73.685827523 kJ/mol      |difference| = 0.000e+00
```

A constant-tau diagnostic path — the same tau every update, with the coordinates propagating —
gives zero work to 1e-6.

### The tiny CUDA run

Two paths, 40 updates over 40 steps, 21 observations:

```
[AIS] 2 independent path(s), tau 0.5 -> 0.0, 40 switching update(s) over 40 steps, 21 observations
[AIS] source cMD/whole_system.dcd at tau 0.5, 20 eligible frame(s) in [0.02, 0.4] ps (inclusive)
[AIS] platform CUDA, device(s) [0, 1], 2 path(s) at a time
[AIS 0000] source frame 15 at 0.32 ps, seeds integrator=1534646695 velocity=1420193027, gpu 0
[AIS 0001] source frame  6 at 0.14 ps, seeds integrator=1607140304 velocity=1492686636, gpu 1
[AIS 0000] 0 barostat(s) in the switching System (fixed volume; no pV work is included)
[AIS 0000] observation 20: tau 0, cumulative work -101.044022 kJ/mol (reduced -40.509341)
[AIS 0001] observation 20: tau 0, cumulative work  -90.856396 kJ/mol (reduced -36.425042)
```

```
trajectory_0000/observations.dcd: 21 frames, 193 atoms | CSV rows: 21
trajectory_0001/observations.dcd: 21 frames, 193 atoms | CSV rows: 21
```

First and last rows of `trajectory_0000`:

```
idx obs frame src_frame src_t  step  t_ps  tau   s     sqrt_s  dW        W          betaW
0   0   0     15        0.32   0     0.0   0.5   0.25  0.5     0.0       0.0        0.0
0   20  20    15        0.32   40    0.08  0.0   1.0   1.0     -6.043509 -101.044022 -40.509341
```

The cumulative column is the running sum of the incremental one, row by row, and the last row
matches `resolved_run.yaml`'s reported total to 1e-6.

---

## 5. Limitations

1. **Forward only.** `tau_start -> tau_end`. The configuration is shaped so a reverse path can be
   added, but no bidirectional workflow exists.
2. **No mid-path restart.** A completed path is skipped, identified by its own completion record and
   its observation count. An incomplete one is **replaced**, never appended to. There is no
   mid-switch checkpoint.
3. **Fixed volume, no pV work.** An NPT source ensemble may seed these paths; the pressure-volume
   contribution to the work is not part of this implementation.
4. **No free-energy estimator.** The work columns are the input a Hummer-Szabo or Jarzynski analysis
   would consume. Nothing here computes one, and nothing here claims a free energy.
5. **DCD carries no velocities**, so every path draws fresh Maxwell-Boltzmann momenta. Two paths
   from the same source frame would differ only in their momenta; sampling without replacement is
   the default for that reason.
6. **Whole-system coordinates only** in this first implementation. A solute-only observation scope
   is refused rather than approximated.
7. **The tiny CUDA run is a correctness test, not a scientific path.** 40 steps of switching is
   chosen so the arithmetic divides exactly, and its work values mean nothing physically.
8. **The acceptance environment lives in a scratch directory.** It is reproducible from the recorded
   command, but it is not the user's working environment, which still carries no `mdtraj`. Running
   AIS there needs `mdtraj` installed or a rebuild through `md-template install`.
9. **`docs/implementation/explicit_solvent/`** still describes the pre-`ca29fcd` architecture and
   carries its HISTORICAL banner. It was not rewritten.
