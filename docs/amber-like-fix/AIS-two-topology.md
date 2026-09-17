# AIS as a transformation between two topologies — design for 0.5.4

2026-09-16. **Status: SIGNED OFF and IMPLEMENTED on `ais/two-topology`** (the decisions are §11). It replaces three
scientific invariants in `CLAUDE.md`; the replacement text is in §8, and no code changes until it
is agreed. Supersedes `docs/backlog.md` entry 15 as the plan; `AIS.md` in this directory remains
the prior analysis (its dispersion-tail fast path becomes moot, §6).

## 1. Decisions already taken

| question | decision |
|---|---|
| what may V0 and V1 differ in | **parameters only** — identical particles, same order, same masses, same constraints. No atom mapping, no softcore, no dummy atoms. Amber's counterpart is sander's no-softcore linear mixing (Amber26 §27.1, Eq. 27.3, p.554) |
| today's single-topology tau AIS | **replaced**. A REST2 tau switch is one choice of end states. REST2, rREST2 and fixed-tau cMD are untouched |
| how V1 is supplied | **a second serialised System and topology on `md-run`** (§4) |

## 2. The Hamiltonian and the work

```text
V(λ, x) = (1 − λ)·V0(x) + λ·V1(x),        λ ∈ [0, 1]
∂V/∂λ   = V1(x) − V0(x)
```

`λ` is the only public, persisted coordinate. It runs **0 → 1**, the Amber convention, and the
source ensemble is sampled from **V0**. A reverse switch is expressed by exchanging the files, never
by a schedule running downhill — which closes the open direction question in `AIS.md`: the
Jarzynski average is always over the V0 ensemble, by construction.

The work convention is unchanged in form: parameters move at frozen coordinates, then the
configuration propagates.

```text
ΔW_j = V(λ_{j+1}, x_j) − V(λ_j, x_j) = (λ_{j+1} − λ_j) · [V1(x_j) − V0(x_j)]
```

Because V is exactly **linear** in λ, the right-hand equality is an identity, not an approximation.
It is also exactly Amber's Eq. 27.30, `(∂U/∂λ)·Δλ`: the O(Δλ²) difference between Amber's
gradient convention and ours, recorded in `AIS.md` §1, **vanishes for linear mixing**. We keep the
finite-difference definition as the contract (so a later non-linear schedule cannot silently
change what `work` means), and record both.

Observation 0 precedes all work and is exactly zero. Switching is at fixed volume; a barostat in
either System is refused.

## 3. The two probes, restated

The present three-point quadratic basis (`a ∈ {0, ½, 1}`) becomes a **two-point linear basis**
`{V0, V1}`, and both values come from ONE evaluation (§5), so the probe costs nothing extra.

* **Work-basis probe** — at the frozen pre-switch `x_j`: `V0(x_j)`, `V1(x_j)`. Defines `ΔW_j`.
* **Observation-potential probe** — at the coordinate a row SAVES, under that row's λ: `V0`, `V1`,
  and the direct `V(λ)`. This is what Hummer–Szabo reweighting consumes.

They remain different coordinates and must never share column names. A row with no saved
coordinate leaves its potentials EMPTY. `potential_v0_kj_mol`, `potential_v1_kj_mol` and
`potential_direct_kj_mol` are written, so `(1−λ)V0 + λV1 = V(λ)` is checkable from the file; at
run time `V1 − V0` from the derivative kernel is checked against the collective-variable values
before a row is written. (As implemented: a separate "reconstructed" column would be the same
arithmetic twice, so it was not added.)

`work_measurement: components` and `verify_every_updates` lose their reason to exist: the
decomposition is no longer a separate, more expensive mode. **Proposal: retire both keys** with a
migration message; every path writes the two-state columns.

## 4. Command surface

```text
-p   V0 topology (PDB)        -s   V0 serialised System        (unchanged meaning: the sampled state)
-p2  V1 topology (PDB)        -s2  V1 serialised System        (new; AIS only)
```

`-s2`/`-p2` have no Amber counterpart (pmemd merges both states into one prmtop; sander passes two
prmtops through a groupfile). They are refused BY NAME on every other protocol, in the preflight,
as the flags-of-another-protocol rule requires. The names follow sander's reading of a second
group (decided 2026-09-16).

`build-md` for `protocol: AIS` gains an `ais_end_states` block naming both files, and `run.sh`
passes them. `ais.tau_start` / `ais.tau_end` and the `&AIS` keys of the same names are **refused
with a migration** naming the replacement. A schedule is `lambda_start: 0`, `lambda_end: 1`, not
configurable in 0.5.4 (a partial window has no use without softcore; refusing is cheaper than
supporting it wrongly). `INTERPOLATION = "linear"` stays the only schedule; smoothstep is out of
scope (§9).

### Where V0 comes from for a REST2 tau switch

Decided 2026-09-16 with the REST2-scaler design (`REST2-scaler.md`): scaled states are **saved by
`build-top`**, per method, as `build/<method>/system_state<n>.xml`, each directory described by
`build/<method>/scaler.yaml` — for REST2, `build/REST2/system_state<n>.xml` (user, 2026-09-16). The
hot-cMD run integrates one of those state files; that same file is AIS's V0, and `build/built.xml`
is V1. A fixed-tau stage writes no System of its own.

AIS calls the scaler's identity function on `-s` and `-s2`. It returns
`{record, method, state, tau, system_sha256}` or `None`; the result is recorded in `AIS_run.json`
so the run names what V0 and V1 are. `None` is legitimate — AIS accepts any parameter-only pair —
and is recorded as such. The source ensemble is tied to V0 by the digest the hot-cMD stage records
for its `-s` (§7). Using a scaled file as an input is accepted; only a request to scale it again is
refused, and that refusal belongs to the scaler, not to AIS.

## 5. Implementation in OpenMM — measured feasible

One mixed System, built at run time from V0 and V1:

1. Pair forces by index. Refuse unequal force counts or classes (§7).
2. A force whose V0 and V1 serialisations are identical (force group normalised) is added ONCE,
   unscaled — `(1−λ)F + λF = F` exactly. Bonds, angles, CMMotionRemover and, for a REST2 pair,
   everything but `NonbondedForce` and `PeriodicTorsionForce` fall here.
3. Every differing pair becomes two collective variables of one `CustomCVForce`,
   `(1-lambda_ais)*U0 + lambda_ais*U1`, with `addEnergyParameterDerivative("lambda_ais")`.
4. A λ update is `context.setParameter("lambda_ais", λ)` — a global, no re-upload, for every
   System, explicit solvent included. `getParameterDerivatives()` returns `V1 − V0` of the
   differing terms, which is `ΔW_j / Δλ` directly; `getCollectiveVariableValues()` returns V0 and
   V1 separately for the probes.

Probe (a one-off script, not a test): alanine dipeptide, amber14 + TIP3P, PME, HBonds; V1 = the REST2
System at tau 0.5. Differing forces: `NonbondedForce`, `PeriodicTorsionForce`.

| check | Reference (double) | CUDA (double) |
|---|---|---|
| `V_mixed(λ) − [(1−λ)V0 + λV1]`, λ ∈ {0, ¼, ½, ¾, 1} | 0.0 | ≤ 1.0e-6 kJ/mol |
| max force deviation | 1.4e-12 | 2.1e-6 kJ/mol/nm |
| `∂V/∂λ − (V1 − V0)` | 0.0 | 5.4e-7 kJ/mol |

Cost, CUDA mixed precision, RTX 3080, 6232 particles, λ updated every step:

| path | ms / step |
|---|---|
| plain V0 dynamics | 0.174 |
| **today's tau AIS**, work mode (energy, `set_tau` re-upload, energy, step) | **7.77** |
| **two-topology**, derivative read + `setParameter` + step | **1.40** |
| two-topology, dynamics only | 0.483 |

So the redesign is ~5.5× faster per update than today's explicit-solvent path, and ~2.8× slower
than plain dynamics because the differing `NonbondedForce` is evaluated twice (as Amber computes
the reciprocal sum twice, p.552). The loop in this probe is a stand-in for `run_one_path`, not
it; the real figure is a benchmark in the test plan. Splitting the solvent–solvent nonbonded out
of the CV is a later optimisation, not 0.5.4.

**Checked 2026-09-16 — a `CustomCVForce` breaks bitwise resume on CUDA.** Same 712-particle
system, RTX 3080, a checkpoint taken 500 steps in, then 1000 steps after loading it:

| System, precision | forces right after load, two loads | positions after 1000 steps |
|---|---|---|
| plain V0, mixed | identical | **identical (0.0)** |
| mixed-state (`CustomCVForce`), mixed | differ by 1.5e-4 kJ/mol/nm | differ by ~0.5 nm |
| mixed-state, mixed, inner Context also checkpointed | differ by 1.5e-4 | differ by ~0.5 nm |
| mixed-state, mixed, `DeterministicForces=true` | differ | differ by ~0.02 nm |
| mixed-state, double | differ by 2.3e-9 | differ by ~0.2 nm |

A plain Context restores bit-for-bit, because the checkpoint carries its CUDA atom ordering. The
inner Contexts of a `CustomCVForce` keep ordering state that no checkpoint captures, so a force
differs in the last bits the moment a run resumes, and chaos grows it. Double precision does not
fix it and costs ~13× (5.9 against 0.46 ms/step at 6232 particles). A resumed path is still a
correct sample of the same switching process, but it is **not the same trajectory** — and
`tests/test_cv_mpi_cuda_ais.py::test_interruption_and_resume_under_mpi_reproduce_every_output`
asserts row-for-row equality with an uninterrupted run. See §11 for the decision this forces.

## 6. What retires, what is reused

**Retires from the AIS path** (with named migrations where a user can meet them):
`TauSwitcher` as AIS's switch; `ais/decomposition.py`'s `rest2-lambda-basis` schema (added to
`HISTORICAL_SCHEMAS`, refused on resume); `ais.tau_start`, `ais.tau_end`, `work_measurement`,
`verify_every_updates`; the NetCDF `tau` attribute check on the source; the columns `tau`,
`tau_before`, `tau_after`, `delta_work_{non_scaled,sqrt_scaled,lin_scaled}_kj_mol` and the
three-group potentials. `global_switching_refusal`, `reparameterise_for_global_switching` and the
dispersion-tail proposal in `AIS.md` lose their only consumer; `TauSwitcher` stays exported only
if something outside AIS uses it (nothing does today — **decide: delete or keep as public API**).

**The omega exclusion leaves the AIS path, deliberately.** Since `7a4ea5d`, `_prepare_ais` calls
`omega_exclusions()`, which refuses an unclassifiable amide candidate instead of scaling it. That
question is "which torsions keep their barrier when the solute is scaled", and under this design
AIS scales nothing: it mixes two finished Systems. For a REST2 tau switch the scaling happens in
the fixed-tau cMD stage whose System becomes V0, and `_prepare_stage` already enforces
`omega_exclusions()` there. So the refusal still guards the Hamiltonian where it is built, and the
AIS preflight loses the call along with `TauSwitcher`. A V0/V1 pair built some other way carries
whatever torsions its builder chose. That is the builder's responsibility, and the
digest check in §7 is what ties the source ensemble to it.
`tests/test_omega_unclassified_is_refused.py::test_no_scaling_surface_calls_the_classifier_directly`
is unaffected.

**Must not change:** `rest2/hamiltonian.py`, `build_scaled_system`, `check_scaling_plan` for
fixed-tau cMD and ladders, the NetCDF `tau` attribute for stages and ladders.

**Reused unchanged apart from field names:** the checkpoint generation transaction, completion
manifests and publication, directory identity and dispositions, MPI coordination and
`paths_for_rank`, the preflight shell, source-frame selection and streaming, velocity resampling,
the step arithmetic of `switching_schedule` and its four cadences, CV reporting and cost
accounting, `write_work_table` assembly.

**Renamed columns** (as implemented): `lambda` for `tau`; `lambda_before`/`lambda_after`;
`potential_v0_kj_mol`, `potential_v1_kj_mol`, `potential_direct_kj_mol`; schema
`two-state-linear` v1; run identity schema v2.

## 7. Preflight refusals — all before any output

Every one of these is a defect that runs to completion and produces a plausible work table:

* particle count, or any **mass**, differs (Eq. 27.30 assumes λ-independent masses; HMR must match)
* constraints, virtual sites, or exclusion lists differ (Amber p.555 note 6: same SHAKE bonds)
* force count or force class sequence differs, or a nonbonded method, cutoff, switching distance,
  PME parameters or periodicity differs
* either System carries a barostat
* topologies disagree on atom names, residues or order, or either disagrees with its own System
* the two Systems are identical (zero work by construction — the current `tau_start == tau_end`
  refusal, restated)
* the source ensemble was not sampled from V0: its recorded System digest (§4) differs from
  `sha256(-s)`. A source with no recorded digest (DCD, a foreign trajectory) is ASSERTED, and the
  run record says so rather than claiming verification — this also retires the hard-coded
  `tau_verified_from_file: False` found at `ais/run.py:2560`

The directory identity (`AIS_run.json`) binds both Systems' and both topologies' digests; a second
invocation with either changed is refused by name.

## 8. The invariants as they would read — FOR SIGN-OFF

Replacing the **AIS** bullet:

> **AIS**: a transformation between two end-state Systems with identical particles, masses and
> constraints, `V(λ) = (1−λ)V0 + λV1`. `λ` is the only public, persisted coordinate and runs 0 → 1;
> the source ensemble is V0's and its digest is checked. Work is `ΔW_j = V(λ_{j+1}, x_j) −
> V(λ_j, x_j)`: parameters move at frozen coordinates, then the configuration propagates.
> Observation 0 precedes all work and has exactly zero. Switching is at fixed volume; a barostat
> in either System is refused.

Replacing **"The potential is exactly quadratic in `a = 1 − τ`, and there are TWO probes of it"**:

> **The potential is exactly linear in λ, and there are TWO probes of it.** `V(λ, x) = (1−λ)V0(x)
> + λV1(x)` is an identity at every coordinate, PME and dispersion correction included, and V0 and
> V1 are read in one evaluation. The work-basis probe runs at the frozen pre-switch `x_j`; the
> observation-potential probe at the coordinate a row saves, under that row's λ. [the remainder of
> the present bullet — never confuse them, empty rather than borrowed, `AIS_hs.csv`, both totals
> written — carries over verbatim]

**"ONE REST2 scaler serves fixed-τ cMD, REST2, rREST2 and AIS"** becomes "fixed-τ cMD, REST2 and
rREST2". The **output-directory identity** bullet replaces "the tau schedule" with "both end-state
digests and the λ schedule". The CV bullets replace `tau` by `λ` where they name AIS.

## 9. Out of scope for 0.5.4, stated so it is not mistaken for an omission

Atom mapping, differing atom counts, softcore (Amber Eq. 27.5–27.7), dummy atoms, `crgmask`,
Boresch restraints, smoothstep schedules (Eq. 27.8), partial λ windows, equilibrium TI/MBAR, and
Crooks reverse paths in one run. Each is a later release; the refusals in §7 are what keep a user
from reaching them by accident.

## 10. Test plan — failing tests first

1. **Hamiltonian identity** (Reference, then CUDA with the feature enabled): mixed energy, forces
   and `∂V/∂λ` equal the linear combination across λ, explicit PME and GBn2.
2. **Work identity**: `ΔW_j` from the derivative equals the direct finite difference; telescoping
   over a path; identical Systems refused; constant λ does zero work.
3. **Every refusal in §7**, each named, each leaving no `-odir`.
4. **REST2 equivalence at the endpoints**: `V(0)` equals the tau-scaled System and `V(1)` equals
   `built.xml`, energy and forces, at several configurations.
5. **Source digest**: AIS accepts a source whose hot-cMD run recorded `sha256(-s)` for its System,
   refuses one that recorded a different digest, and records a digest-less source as ASSERTED.
6. **Checkpoint through a `CustomCVForce`** (CUDA): interrupt mid-path and resume. The restored
   λ, accumulated work, counters, positions and velocities equal the committed generation exactly;
   the completed path is valid and complete. No row-for-row equality past the resume point (§11.8).
7. **Migrations**: `tau_start`, `tau_end`, `work_measurement`, `verify_every_updates` and a resume
   onto a `rest2-lambda-basis` directory refused with the replacement named.
8. **Migrate** the bookkeeping suites (identity, recovery, CV, invocation accounting, MPI+CUDA)
   per the classification in the survey; classify every retired test as obsolete with the deleted
   feature named, never silently removed.
9. **Benchmark** the real `run_one_path` on CUDA against the 0.5.3 path, one explicit and one GBn2
   system.

## 11. Decisions — answered 2026-09-16

1. **Flags are `-s2` / `-p2`.**
2. **V0 is a state file saved by `build-top`** (`build/<method>/system_state<n>.xml`), per the
   REST2-scaler design; the user chose "save states" over a `system.xml` written by the stage (§4).
3. **Confirmed: `work_measurement` and `verify_every_updates` are retired** with a migration message.
4. **`TauSwitcher` is removed** if nothing outside the old AIS uses it (nothing does today), with
   the global-parameter switching code that only it consumes.
5. **§8 accepted: AIS is no longer associated with the REST2 scaler.**
6. **The omega exclusion belongs to the REST2 scaler**, not to AIS (§6).
7. **Linear interpolation between two topology files is the design.** The user's reasons: linear
   in tau made the work and its decomposition hard to handle, and the analytically fitted
   dispersion correction proposed in `AIS.md` is fragile and does not generalise to NpT. Future
   development keeps two-topology switching and adds **non-linear schedules** — which is why the
   finite-difference work definition in §2 stays the contract rather than `(∂V/∂λ)·Δλ`.
8. **Resume is exact in STATE, not in trajectory.** A `CustomCVForce` cannot resume bit-for-bit on
   CUDA (§5). A resume restores the committed generation exactly — λ, accumulated work, counters,
   positions, velocities — and continues the same switching process; the trajectory after that
   point is a new realisation. Tests assert exact restoration at the resume point and a complete,
   valid path; row-for-row equality with an uninterrupted run is dropped for AIS.
9. Related release scope: 0.5.4 is validated on **cMD, hot-cMD (fixed tau), REST2, AIS and
   umbrella**; rREST2 is archived (refused, moved to `archive/rREST2/`, tagged) by this session
   after AIS lands.
