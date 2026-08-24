# Claude Code instruction: OpenMM 8.6.0 and ALA cMD/REST2 ensemble validation

Date: 2026-08-24  
Repository: `csy0000/MD-templates`  
Control branch: `dev`  
Expected starting commit: `39b7f52b7eb1cb60a47856f432605885b15f3d6e`, or a direct descendant

## Outcome

Upgrade the supported and tested OpenMM version to the exact stable release **8.6.0**, repair the
current ensemble/configuration inconsistency, and perform four short alanine-dipeptide validation
campaigns through the repository's public `MD_system_gen.py` and `MD_input_gen.py` workflow:

| test | solvent | replicas | pre-production equilibration | production |
|---|---|---:|---:|---:|
| cMD | explicit ff19SB/OPC | 1 | 1 ns | 5 ns NPT |
| cMD | implicit GBn2/mbondi3 | 1 | 1 ns | 5 ns nonperiodic constant-T |
| REST2 | explicit ff19SB/OPC | 6 | 1 ns independently at each tau, no exchanges | 5 ns per replica, NPT, exchange every 10 ps |
| REST2 | implicit GBn2/mbondi3 | 4 | 1 ns independently at each tau, no exchanges | 5 ns per replica, nonperiodic constant-T, exchange every 10 ps |

The user's phrase "5 ns NPT production" applies only to explicit solvent. An implicit-solvent
System has no periodic volume or pressure, so it must have no barostat, box, PME, salt, or NPT label.
Call it **nonperiodic constant-temperature MD**, not NPT and not a fixed-volume periodic ensemble.

This is a correctness and workflow validation, not a sampling-convergence study. Five nanoseconds
must not be described as validating REST2 mixing or converging the ALA conformational ensemble.

Do not run RGDfV, alchemy, Deeptime/MSM, long production, or any method-development branch in this
task. Preserve the accepted 2026-08-24 OpenMM-8.5.2-tag 5 ns data and journals unchanged.

## Repository and branch law

1. Work directly on `dev`. Do not create a feature branch or PR.
2. Fetch first, confirm the branch and clean worktree, and record the exact starting SHA.
3. If `dev` advanced, integrate it without force-pushing or discarding user work.
4. Read `CLAUDE.md`, the active README/configuration/install documentation, both public generators,
   canonical profiles/schema/resolver, `stage.py`, `runner.py`, `md.py`, `rest2.py`,
   `equilibration.py`, `system.py`, `runstate.py`, current tests, and the 2026-08-24 journal.
5. Make focused commits and push them to `dev`. Commit code/config/test fixes before building new
   bundles. Every new bundle and run must record that exact full 40-character code commit.
6. Do not commit trajectories, checkpoints, serialized States, production logs, environments, or
   other generated simulation data.

## Part 1 — upgrade cleanly to OpenMM 8.6.0

Update every active version constraint and installation path that currently selects OpenMM 8.5.2:

- environment and lock/specification files;
- installation documentation and validation scripts;
- the repository's agent-readable installation skill/instructions, if versioned here;
- support matrix, README, examples, CI, test expectations, and provenance validators;
- generated environment metadata and golden files only where they represent the active version.

Require an exact stable version string of `8.6.0`. Do not accept `8.6.0.dev-*`, a development
snapshot, or merely a commit that resembles the release. Record package manager, package build,
Python version, OpenMM version, CUDA platform/plugin information, driver/runtime, and GPU UUID.

Do not relabel the existing OpenMM `8.5.2.dev-36a30cb` runs. Commit `36a30cb` is the 8.5.2 release
commit, but those results remain the 8.5.2-tag baseline with the version string they actually
reported.

Run the full fast suite and available CPU integration tests under 8.6.0 before any CUDA campaign.
Run environment validation on CUDA and confirm that OpenMM reports the intended devices and mixed
precision.

OpenMM 8.6.0 introduces native multistate sampling, but do **not** replace the custom REST2 backend
in this task. The repository has project-specific tau scaling, omega exclusion, per-stream output,
GPU mapping, and committed-generation restart behavior. Briefly record whether the native sampler
could become a future backend, but do not add a second production implementation or change the
exchange schedule.

## Part 2 — repair configuration and execution-path inconsistencies

The current tree is internally contradictory:

- the active default says production is NPT;
- the newer generated `stage.py` path can run NPT cMD;
- legacy `run_md()` rejects every ensemble except NVT;
- `run_rest2_remd()` rejects NPT;
- `BAROSTAT_STAGES` includes `cMD_1` but not `REST2_1`;
- old comments still call `width/sqrt(2)` the dodecahedral periodic-copy distance, although that
  quantity is the reduced-box cutoff height.

Fix the model rather than adding campaign-only overrides.

### Canonical semantics

- Explicit cMD: `NPT`, 300 K and 1 bar, one `MonteCarloBarostat`.
- Explicit REST2: `NPT`, every replica at the same physical bath temperature and pressure, one
  independently seeded barostat per replica.
- Implicit cMD/REST2: nonperiodic constant-temperature dynamics; no ensemble field that falsely
  implies a physical volume, and no pressure/barostat/periodic settings.
- Explicit and implicit profiles must resolve these facts without relying on a global default that
  is invalid for one of them.
- Invalid combinations must be rejected during config validation, before a System or run directory
  is created.

Make `MD_system_gen.py -> MD_input_gen.py -> generated stage launcher -> package runner` the single
public behavior. Reconcile the older `md.py/rest2.py` backend with that behavior or clearly route
the public workflow through one implementation; do not retain two executable paths with different
ensemble semantics.

Correct active README/configuration comments:

- shortest lattice translation is `width` for the OpenMM cube, dodecahedron, and octahedron;
- `width/sqrt(2)` for a dodecahedron is the reduced-box height used for cutoff legality, not the
  solute-copy spacing;
- the repository does support implicit GBn2/mbondi3;
- 2.0 nm is the repository's chosen explicit default in OpenMM padding semantics.

Clarify journal provenance by distinguishing the code commit used to run simulations from later
documentation commits.

## Part 3 — NPT Hamiltonian replica exchange

Implement explicit-solvent NPT REST2 as Hamiltonian replica exchange at one physical temperature and
one pressure.

For state `k`, use the general configurational reduced potential

```
u_k(x, V) = beta_k * (U_k(x; V) + p_k * V)
```

and compute the pair exchange log probability from all four reduced potentials:

```
log_accept = -[u_i(x_j,V_j) + u_j(x_i,V_i)
               - u_i(x_i,V_i) - u_j(x_j,V_j)]
```

For this REST2 protocol, all replicas have the same physical `beta` and `p`, so the `pV` terms
cancel algebraically. Keep the general implementation and tests rather than relying on that
cancellation silently. The effective solute temperature is an interpretation of Hamiltonian
scaling, not a different thermostat temperature.

At an exchange:

- cross-evaluate each configuration with its own periodic box under both Hamiltonians;
- on acceptance, exchange positions, box vectors, and velocities as one complete sampler state;
- because physical temperatures are identical, do not rescale velocities;
- on rejection, restore both replicas exactly;
- keep each Hamiltonian/tau tied to its Context; walkers move between tau states;
- preserve the even/odd neighboring schedule and durable walker-to-state history;
- record `U`, `pV`, full reduced potentials, `log_accept`, decision, pair, round, tau, walker
  mapping, volumes, and RNG provenance.

Each explicit REST2 replica must contain exactly one active `MonteCarloBarostat` at 1 bar and 300 K,
with independently derived recorded seeds. Volume changes must be allowed during both tau-specific
equilibration and REST2 production.

Do not assume that adding a barostat is enough. Test detailed-balance arithmetic independently,
including different boxes and cross-Hamiltonian energies.

## Part 4 — implicit REST2 definition

Implicit REST2 is nonperiodic and has no `pV` contribution. Use the validated ff19SB +
GBn2/mbondi3 ParmEd path and require the whole alanine solute to be the enhanced region.

Preserve the current tau convention:

```
solute-solute scale       = (1 - tau)^2
solute-environment scale  = (1 - tau)
s                          = (1 - tau)^2
```

For a whole-system implicit calculation, scale the complete `CustomGBForce` energy by `s`,
including its charge-dependent and nonpolar terms, as the current implementation intends. Continue
the standard REST2 handling of bonded terms: do not newly scale bonds or angles. Audit every force
class in the generated implicit System and refuse unknown unclassified energy-bearing forces rather
than silently leaving them at the wrong scale.

The tau-zero implicit System must be energy- and force-identical to the unscaled GBn2/mbondi3 System
within tight numerical tolerance. The implicit System must remain nonperiodic, `NoCutoff`, and
contain zero barostats, water molecules, and ions at every tau.

## Part 5 — exact ALA protocols

Use the repository's accepted capped alanine-dipeptide input and stereochemistry.

### Shared tau convention

Use the active default `tau_min = 0`, `tau_max = 0.5`, and linear interpolation.

Explicit, 6 replicas:

```
tau = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
s   = [(1-tau)^2] = [1.0, 0.81, 0.64, 0.49, 0.36, 0.25]
```

Implicit, 4 replicas:

```
tau = [0.0, 1/6, 1/3, 0.5]
s   = [(1-tau)^2] = [1.0, 25/36, 4/9, 0.25]
```

Write the exact floating-point values and formulas to resolved configuration and provenance.

### A. Explicit cMD

- ff19SB + OPC and compatible ions;
- dodecahedron, 2.0 nm OpenMM padding semantics;
- 0.15 M NaCl, PME, 1.0 nm cutoff, 0.10 nm cutoff-height margin;
- HMR 3.024 amu, HBonds, 4 fs Langevin-middle, 300 K, CUDA mixed precision;
- 1 ns staged equilibration: 250 ps NVT, 250 ps restrained NPT, 500 ps free NPT;
- 5 ns unrestrained NPT production at 1 bar;
- all atoms every 100 ps, solute every 10 ps, state every 100 ps.

### B. Implicit cMD

- ff19SB + GBn2/mbondi3;
- nonperiodic, NoCutoff, no solvent/ions/box/barostat;
- no HMR, HBonds, 2 fs Langevin-middle, 300 K, CUDA mixed precision;
- total 1 ns equilibration: preserve the accepted initial 20 ps positional-restraint phase and add
  980 ps unrestrained constant-temperature equilibration;
- 5 ns unrestrained nonperiodic constant-temperature production;
- all atoms every 100 ps and solute every 10 ps. The selections are identical; record that fact.

### C. Explicit NPT REST2

- same explicit Hamiltonian and numerical settings as explicit cMD;
- six tau replicas;
- begin from one accepted physical-system prepared state, then assign distinct velocities/seeds;
- equilibrate each replica for exactly 1 ns under its own tau Hamiltonian at 300 K and 1 bar;
- no exchanges and no production reporters during this tau-specific equilibration;
- save a separate equilibration endpoint and QC record for every tau;
- then run exactly 5 ns of committed production per replica;
- exchange every 10 ps: exactly 500 exchange rounds;
- even rounds propose (0,1), (2,3), (4,5); odd rounds propose (1,2), (3,4);
- exactly 1,250 pair attempts over 500 rounds;
- all atoms every 100 ps and solute every 10 ps per replica.

### D. Implicit REST2

- same implicit Hamiltonian and numerical settings as implicit cMD;
- four tau replicas;
- distinct velocities/seeds;
- equilibrate each replica for exactly 1 ns under its own tau Hamiltonian, with no exchange and no
  production reporters;
- then run exactly 5 ns of committed production per replica;
- exchange every 10 ps: exactly 500 exchange rounds;
- even rounds propose (0,1), (2,3); odd rounds propose (1,2);
- exactly 750 pair attempts over 500 rounds;
- all atoms every 100 ps and solute every 10 ps per replica, with identical-selection provenance.

Equilibration time is not production and must never enter committed 5 ns frame/time counts.

## Part 6 — tests required before CUDA execution

Add focused tests that independently establish:

1. exact OpenMM `8.6.0` version validation and provenance;
2. explicit/implicit canonical config validation and rejection of impossible cross-mode settings;
3. both public cMD entry paths agree on NPT explicit and nonperiodic implicit semantics;
4. explicit NPT cMD has exactly one correctly parameterized barostat;
5. every explicit REST2 replica has exactly one independently seeded barostat;
6. implicit cMD/REST2 has no periodic force, box-dependent nonbonded method, or barostat;
7. NPT exchange reduced-potential arithmetic matches an independent hand calculation;
8. at common temperature/pressure the `pV` cancellation matches the four-energy expression;
9. accepted exchange swaps positions/box/velocities and rejected exchange restores all three;
10. tau ladders and Amber-style `(1-tau)^2/(1-tau)` scaling are exact;
11. tau-zero explicit and implicit Hamiltonians reproduce the unscaled energy and forces;
12. implicit `CustomGBForce` complete-energy scaling and whole-system-selection refusal work;
13. 1 ns tau-specific equilibration is separate from production and contains zero exchange attempts;
14. 5 ns / 10 ps gives 500 rounds, with 1,250 explicit and 750 implicit pair attempts;
15. reporting intervals, committed-generation continuation, log append, RNG restoration, and
    walker/state reconstruction remain correct;
16. a short CPU or Reference-platform two-replica test exercises one accepted and one rejected
    exchange without relying only on mocks.

Run the full fast suite and all relevant CPU integration/slow tests. Do not launch CUDA runs while
any failure is unexplained. Fix code or tests; do not weaken scientific assertions to obtain green.

## Part 7 — CUDA execution and scheduling

Use a new data root outside Git, for example:

```
/path/to/MD-analysis-data/20260824_openmm860_ala_validation/
  cmd/explicit/
  cmd/implicit/
  rest2/explicit/
  rest2/implicit/
  configs/
  logs/
```

If occupied, choose a non-destructive sibling and record it. Never overwrite the earlier campaign.

Before launch inspect all nine GPUs: index, UUID, model, memory, utilization, and compute processes.
Set `CUDA_DEVICE_ORDER=PCI_BUS_ID`; use stable UUID mapping and exclusive locks; never kill or share
with unrelated work.

Safe schedule:

1. run explicit and implicit cMD concurrently on two idle GPUs;
2. run the six-replica explicit REST2 validation on six idle GPUs;
3. run the four-replica implicit REST2 validation after resources are free.

Do not run ten REST2 replicas simultaneously on nine GPUs for this validation. One Context per
selected physical GPU is the intended test. Record the process/GPU/context mapping and verify that
each GPU actually receives work.

Use durable tmux or an equally inspectable supervisor. Maintain machine-readable status and exact
monitor/stop/resume commands. A single failed protocol must not stop unrelated safe tests; diagnose
it while the others continue.

## Part 8 — runtime validation and acceptance

For every run verify:

- exact OpenMM 8.6.0 and exact repository/bundle/config hashes;
- requested versus realized steps, time, frames, rounds, and pair attempts;
- finite potential/kinetic energy, temperature, positions, velocities, and forces;
- no production restraints;
- continuation watermarks and no duplicate or uncommitted trajectory tail.

Additionally for explicit runs:

- box vectors and volume finite;
- volume changes under NPT;
- exactly one active barostat per Context;
- reduced-box height remains above twice the 1.0 nm cutoff;
- exact neighboring-image solute distance remains above the cutoff in every saved frame;
- water/ion counts and ff19SB/OPC identity are correct.

Additionally for implicit runs:

- no meaningful periodic box or volume/density observable is reported;
- no barostat, water, ions, PME, or cutoff;
- GBn2/mbondi3 identity and complete-GB scaling provenance are present.

For REST2:

- every tau-specific equilibration reaches 1 ns with zero exchange attempts;
- every replica reaches exactly 5 ns committed production;
- explicit: 500 rounds and 1,250 pair attempts;
- implicit: 500 rounds and 750 pair attempts;
- exchange log is monotonic and reconstructs every walker/state mapping;
- all four cross energies/reduced potentials are finite;
- acceptance values are in [0,1];
- round trips and acceptance may be reported descriptively, but no minimum mixing threshold is a
  pass criterion for a 5 ns plumbing test.

Write `docs/journal/2026-08-24_openmm860_ala_cmd_npt-rest2_validation.md` containing environment,
commits, resolved protocols, tau/scaling tables, seeds, GPU UUIDs, timing, file counts, exchange/QC
tables, failures/retries/fallbacks, and exact reproduction/resume commands.

Return `PASS` only when all four protocols and their stated test gates complete. Return `RUNNING`
with exact monitoring commands if valid jobs remain active. Return `PARTIAL` if a real blocker
remains; do not shorten, relabel, or silently change an ensemble to obtain a pass.
