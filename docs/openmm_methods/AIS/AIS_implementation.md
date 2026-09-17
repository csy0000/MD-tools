# How AIS is implemented in this version

This document describes what the AIS runtime actually does and what each choice rests on. It
complements the README beside it: the README is how to *use* AIS (inputs, configuration fields,
generated files, output columns); this is how it *works*. The design record, with the decisions and
the measurements behind them, is `docs/amber-like-fix/AIS-two-topology.md`.

Nothing here is aspirational. Every number quoted was measured, on the system and device named
with it.

---

## 1. What a switching path is

```text
V(λ, x) = (1 − λ)·V0(x) + λ·V1(x),        λ: 0 → 1
```

A path starts from a frame of an equilibrium ensemble of **V0**, draws fresh Maxwell–Boltzmann
velocities with its own seed, and integrates while λ moves linearly to 1. V0 is `-s`/`-p`; V1 is
`-s2`/`-p2`. The temperature never changes and the volume is fixed.

Until 0.5.4 AIS switched ONE System along a REST2 τ. That made the potential quadratic in
`a = 1 − τ`, needed a three-point basis probe to decompose the work, and could only switch explicit
solvent cheaply with an analytic correction for the dispersion tail that does not survive a
changing volume. Two files and a straight line between them need none of it.

## 2. The pair

`md_tools.ais.two_state.pair_plan(V0, V1)` accepts a pair only if it is parameters-only, and names
every problem at once otherwise:

| checked | why |
|---|---|
| particle count, masses, constraints, virtual sites, default box | the work convention assumes the same particles and λ-independent masses; HMR must match |
| a barostat in either | switching is at fixed volume |
| the force list, class by class, in order | forces are paired by position |
| nonbonded method, cutoff, switching, dispersion correction, Ewald tolerance, PME parameters, exception pairs (and exclusions of custom nonbonded/GB forces) | these are not parameters: differing values are two treatments of interactions, not two strengths |
| a global parameter already named `ais_lambda` | it would collide with the switch |
| every force identical | zero work by construction |

`-p` and `-p2` must also agree atom by atom on name, residue name and residue index.

## 3. The mixture

`TwoStateHamiltonian` builds one System:

1. a force whose V0 and V1 serialisations are identical (force group normalised) is added once,
   unscaled — `(1 − λ)F + λF = F` exactly;
2. every differing pair becomes two collective variables of ONE `CustomCVForce`,
   `(1 − ais_lambda)·end0 + ais_lambda·end1`, with an energy-parameter derivative on `ais_lambda`.

Moving λ is `Context.setParameter`. Nothing is re-uploaded, for any System. For a REST2 pair only
the `NonbondedForce` and `PeriodicTorsionForce` differ; bonds, angles and the centre-of-mass motion
remover are shared.

## 4. The work, and the two probes

```text
ΔW_j = V(λ_{j+1}, x_j) − V(λ_j, x_j) = (λ_{j+1} − λ_j)·(V1 − V0)(x_j)
```

`dV/dλ = V1 − V0` does not depend on λ, so the work of a switch is one
`getState(getParameterDerivatives=True)` at the frozen pre-switch coordinate, taken before λ moves.
The equality with the finite difference is exact for a linear path, and
`tests/test_ais_two_state.py` checks it against two direct energies.

The **observation probe** runs at the coordinate a row SAVES: one `getState` for `V(λ)` and
`dV/dλ`, from which `V0 = V − λ·dV/dλ` and `V1 = V + (1 − λ)·dV/dλ` follow exactly. `dV/dλ` is then
checked against an independent measurement — the collective-variable values, evaluated by the inner
Contexts rather than by the derivative kernel — within a precision-dependent tolerance, and the row
is refused if they disagree. A row that saved no coordinate has empty potentials.

## 5. NVT only, and the refusal that enforces it

A pressure–volume term in the work would make the path measure something the Jarzynski and Crooks
relations are not written for. The preflight refuses a barostat in either end state before the run
directory exists — no `-odir`, no `resolved.config`, no `.out`, no `.log`.

## 6. The schedule

Every length is an **integer step count**. Three counts must divide exactly, and each failure is
refused with the arithmetic that would fix it rather than rounded away:

```text
switching_steps % parameter_update_interval_steps == 0
observation_interval_steps % parameter_update_interval_steps == 0
switching_steps % observation_interval_steps == 0
```

**Four independent cadences**, because they answer four different questions:

| field | question |
|---|---|
| `observation_interval_steps` | how often the WORK is measured — this one is the method |
| `trajectory_interval_steps` | how often a configuration is written to the path's NetCDF |
| `state_interval_steps` | how often the thermodynamic state is tabulated |
| `checkpoint_interval_steps` | how often the path becomes resumable |

Each divides `switching_steps` on its own, so every stream has a record on the final step. `0`
disables the state table and the checkpoint — never the observations.

λ runs 0 → 1 and nothing else: the source ensemble is V0's by construction, so the Jarzynski
average is over V0. A reverse switch is a second campaign with the files exchanged.

## 7. Identity, resume and cost

**An output directory has one identity.** `AIS_run.json` (schema v2) records both end-state
digests, the source digest, the λ schedule, the seed policy, the selected frames, the reporting
schema, the path count, the column schema and the AIS schema `two-state-linear/v1`. A second
invocation into the same `-odir` with any of those changed is refused by name. A v1 identity — the
single-topology AIS — is refused as such.

**The source is verified when it can be.** A stage's whole-system AMBER NetCDF records
`system_sha256` when the Hamiltonian it integrated is its `-s` unmodified (no scaling built in
memory, no restraint, no umbrella bias). AIS refuses a source whose recorded digest is not V0's,
and records a source that states none as ASSERTED.

**Path identity does not depend on the worker count.** `paths_for_rank(rank, size, total)` is a
pure function, and global path *n* always writes `AIS_traj000n.nc`.

**An interrupted path resumes mid-path**, from its last committed generation: new generation,
fsync, digest, sidecar, fsync, then `current_checkpoint.json` replaced atomically last. The
committed state carries λ, the accumulated work, the counters, the stream counts and the AIS schema;
a checkpoint from the single-topology AIS is refused.

**Resume is exact in state, not in trajectory, on CUDA.** A plain OpenMM Context restores
bit-for-bit, because the checkpoint carries its CUDA atom ordering. The inner Contexts of a
`CustomCVForce` keep ordering state no checkpoint captures, so forces differ in their last bits the
moment a run resumes and the trajectory diverges from an uninterrupted one. Measured on a
712-particle explicit system: two loads of one checkpoint differ by 1.5e-4 kJ/mol/nm in force and
~0.5 nm in position after 1000 steps; checkpointing the inner Context too, `DeterministicForces`,
and double precision (13× slower) did not make it exact. On CPU and Reference a resumed path
reproduces an uninterrupted one exactly.

## 8. Measured cost, for reference

Alanine dipeptide in TIP3P, 6232 particles, PME, CUDA mixed precision, RTX 3080, λ updated every
step. V1 = the REST2 System at τ = 0.5.

| path | ms / step |
|---|---|
| plain V0 dynamics | 0.174 |
| the retired tau switch, `work` mode (energy, re-upload, energy, step) | 7.77 |
| two-state: derivative read + `setParameter` + step | 1.40 |
| two-state, dynamics only | 0.483 |

A switch is ~5.5× cheaper than the tau switch was. Dynamics cost ~2.8× plain V0 because the
differing `NonbondedForce` is evaluated for both end states — as Amber computes the reciprocal sum
twice (Amber 2026 manual, p. 552). These loops stand in for `run_one_path`; they are not it.

---

## What this version does not claim

* The costs above are for one system, one GPU and mixed precision.
* Evaluating only the solute-dependent part of the nonbonded force twice — splitting the
  solvent–solvent interactions out of the mixture — is a possible optimisation and is not done.
* Non-linear schedules, softcore and atom mapping are future work, not partial features.
