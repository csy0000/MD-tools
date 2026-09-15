# REST2: the Amber-like output fixes

2026-09-14. Branch `gpu/0.5.3-evidence`, six commits `b3c3ff6` … `7ed8a5a`. **Not pushed, not
released.**

The starting question was how Amber's `.out` compares with ours, and the answer was that an Amber
`mdout` is very nearly enough to reconstruct the run while ours was not. Everything below follows
from that, plus two defects the comparison exposed.

---

## What changed

| # | Change | Commit |
|---|---|---|
| 1 | `energy_components.csv` decomposes on the explicit route too | `b3c3ff6` |
| 2 | Stage `.out` gains `System`, `Method`, `Selections` | `b3c3ff6` |
| 3 | Ladder `.out` gains a `REST2` block and `TIMINGS` | `48d838c` |
| 4 | PyMBAR's banner no longer lands in `REST2.out` | `00e0035` |
| 5 | `energy_components.csv` is a real decomposition, not a lumped total | `b3c3ff6` |
| 6 | An rms fluctuation over one sample reads `n/a`, not `0` | `b3c3ff6` |
| 7 | Ladder `.out` reports throughput and per-step cost | `48d838c` |
| + | The exchange stops recomputing an energy it already had | `00e0035` |
| + | `u_evaluated` is observed rather than asserted | `00e0035` |
| + | Documented output names that matched no file | `ce6ee88` |
| + | Post-hoc term-by-term decomposition | `a78217d` |
| + | CHANGELOG and release notes | `7ed8a5a` |

---

## 1 & 5. `energy_components.csv` was a decomposition on one build route only

**What was wrong.** OpenMM separates energies only by *force group*, and the two build routes
disagree about them:

| route | force groups in `built.xml` | columns in `energy_components.csv` |
|---|---|---|
| implicit (ParmEd `createSystem`) | `0` bond, `1` angle, `2` torsion, `11` GB+nonbonded | **6** — a real breakdown |
| explicit (OpenMM `ForceField.createSystem`) | `0` only — everything | **3** — one joined column |

Measured on your registered data, not inferred: `ALA-test/cMD-hot-implicit` and
`phenol-test/REST2-implicit` both give `[0, 1, 2, 11]`; `ALA-test/cMD-cold-explicit` and
`phenol-test/cMD-hot-explicit` both give `[0]`. The explicit files carry a single column headed
`CustomExternalForce+HarmonicAngleForce+HarmonicBondForce+MonteCarloBarostat+NonbondedForce+PeriodicTorsionForce`
whose value is bit-identical to the total potential energy in `mdout_*.csv`. **31 such files exist
under `$MD_DATA`.** Nothing in the output distinguished the two cases, which is the worst property
a diagnostic can have.

**Confirmed across the three-combination matrix**, built through the real `build-top` path:

| | ff14SB+GBn2 | ff14SB+TIP3P | ff19SB+TIP3P |
|---|---|---|---|
| System route | ParmEd | OpenMM `ForceField` | OpenMM `ForceField` |
| particles | 22 | 1796 | 1796 |
| force groups | **[0, 1, 2, 11]** | [0] | [0] |
| decomposes as built | **yes** | no | no |
| decomposition probe engages | **no — not needed** | **yes** | **yes** |
| columns | 4, real | 5 via the probe | 5 via the probe |
| CMAP present | no | no | **yes** |

So the defect is **explicit-solvent-only**, on both force fields: GBn2 already decomposes through
ParmEd's grouping and the probe correctly declines to engage, while both explicit builds need it.
ff19SB behaves exactly as ff14SB does here — CMAP adds a force but not a grouping problem.

(ff19SB+TIP3P is an off-spec pairing: ff19SB's CMAPs were fit with OPC. `build-top` says so —
`CROSSED PROTEIN/WATER PAIR ... will build and run; with a water model it was not fit for` — and
records `supported_pair_expects: amber14-all.xml`. It is tested because it is the only shipped way
to get CMAP into an explicit box.)

**Why it was not fixed by regrouping the run's System.** A force group is part of the serialised
System, so reassigning one changes `system_sha256`, invalidates the checkpoint fingerprint of every
run in flight, and makes registered data incomparable with new data — all for a diagnostic.
`stage.py` already said exactly this in a docstring; the fix respects it.

**What changed.** When a System carries no usable groups, `_EnergyDecompositionReporter` probes a
**group-separated copy** on the run's own platform. The production System, `built.xml` and every
digest taken from them are untouched.

**Tested.** On the real all-group-`0` explicit System:

```
force groups in the run's System: [0]        probe engaged: True
HarmonicBondForce                      13.631228
HarmonicAngleForce                     23.089235
PeriodicTorsionForce                   55.987698
NonbondedForce                     458746.732200
NonbondedForce [PME reciprocal]   -566950.034600
SUM                               -108110.594239
total potential energy            -108110.594250      (1.1e-05 kJ/mol, ~1e-10 relative)
run's System still all-group-0: True     built.xml on disk unchanged: True
```

**What it still cannot reach.** `EELEC` vs `VDWAALS`, and the 1-4 terms from either. All **7988**
1-4 pairs are exceptions *inside* the single `NonbondedForce`, which evaluates charge and dispersion
in one kernel. Amber prints them apart because its energy routines are written term by term. That
split needs duplicated forces — see the post-hoc module below.

---

## 2. A stage `.out` now says what ran

**What was wrong.** Amber's `mdout` opens with `1. RESOURCE USE` (a topology census) and
`2. CONTROL DATA FOR THE RUN` (every resolved namelist variable in named groups, including
`cut`, `NFFT1/2/3`, `Ewald Coefficient`, `Interpolation order`, `gamma_ln`, `ig`, `tol`), plus
resolved *selections with match counts* — `Mask :1-3 & !@H=; matches 10 atoms`. 428 lines for a
20 ps run. Ours was 52 lines: inputs, step count, platform. The cutoff, the PME treatment, the
constraint count, the net charge and the restrained-selection size all existed — in
`resolved.config`, the machine record and `solute.yaml` — so answering "what was the cutoff" meant
opening three files.

**What changed.** Three sections before `Run`, all read off the **serialised System** rather than
the configuration that asked for it (a config claiming 1.0 nm and a System carrying 0.8 nm are
different runs, and only one integrates). The same facts also enter the `.log` as structured
fields, because prose is not a database.

**Tested** — real output from a run under this branch:

```
System
  atoms                        7920
  residues                     2641 (HOH 2630, Cl- 4, Na+ 4, ACE 1, ALA 1, NME 1)
  net charge                   -0.000 e
  degrees of freedom           15855
  box                          4.674 x 4.963 x 4.497 nm  volume 104.323 nm^3

Method
  nonbonded                    PME, cutoff 1.0 nm, Ewald tolerance 0.0005, grid chosen per Context
  dispersion correction        on
  switching                    off
  exceptions (1-4 pairs)       7988
  constraints                  7902 bond(s)
  barostat in system           MonteCarloBarostat

Selections
  solute                       22 atom(s) (restraint and REST2 region)
  omega bonds unscaled         not applicable at tau = 0 (nothing is scaled, so nothing is exempted)
```

`15855 = 3×7920 − 7902 − 3` — the DOF count behind every reported temperature, which is why
Amber's step-0 temperature looks wrong until you know it.

**Two bugs I introduced here and then fixed**, both worth knowing because they are the exact failure
mode this work exists to remove:

1. **A fabricated box.** `getDefaultPeriodicBoxVectors` cannot answer "is there a box" — an OpenMM
   System defaults to a **2 nm cube** — so every implicit-solvent stage would have printed
   `box 2.000 x 2.000 x 2.000 nm  volume 8.000 nm^3`, invented, in the same voice as the real
   measurement two lines above. Now keyed off the nonbonded method being periodic, which is what
   makes a box load-bearing. Verified both ways: 41 implicit-solvent tests pass, and a
   `NoCutoff` System reports no box.
2. **`omega bonds unscaled 0: none`** at τ = 0, which invites the reader to conclude the classifier
   found no omega bonds. It found none *applicable*: at τ = 0 nothing is scaled so nothing is
   exempted, while the same system on a ladder excludes two. Now says so.

---

## 3 & 7. A ladder `.out` names the Hamiltonian it integrated

**Tested** — real output, 8-state ladder:

```
# REST2:
#   tau ladder            0, 0.071429, …, 0.5   (8 state(s), one temperature 300.0 K, NVT)
#   scaling               solute-solute (1-tau)^2, solute-environment 1-tau
#   left unscaled         bonds unscaled, angles unscaled, ordinary amide omega unscaled
#   solute region         22 atom(s), 2 omega bond(s) excluded
#   system sha256         7521d7a01b63e5762e4650ff4a807b9d8a78ae167dda8eab5b96f4c30ed9e6f5
#   velocities on swap    never rescaled (one beta across the ladder)
# TIMINGS:
#   elapsed               2.1 s
#   throughput            32.25 ns/day per replica, 258.01 ns/day aggregate over 8 state(s)
#   per step              5.3579 ms
```

**Why this matters more than it looks.** Amber prints the equivalent REAF block — τ, the
`gti_add_re` scheme, the mask and its matched atom count — and **cannot name the resulting
Hamiltonian**, because `gti_add_re=6` is an index into a table in the manual and the scaled
potential exists only inside the running binary. A rung here *is* a serialised System, so it can be
named and digested. That is the one place this package is structurally ahead of Amber, and it was
not being said.

---

## 4. PyMBAR's advice was in your provenance file

`REST2.out` contained PyMBAR's timeseries caveat and the JAX 64-bit banner. Cause:
`environment_versions()` did `import openmmtools` **purely to read its version**; that import pulls
in PyMBAR, which writes both to **stderr**; and `executor.py:835` deliberately redirects stdout
*and* stderr into the run's `.out` (which is how a real runtime failure reaches the report file
that is then checked for the completion marker). Fixed by reading the version from distribution
metadata instead of importing the package — same fact, `0.26.0`, **0 bytes** on either stream.

---

## The exchange no longer recomputes an energy it already had

`u[i][state_to_walker[i]]` is the energy of the configuration state *i*'s Context is already
holding — `_gather_configurations` read it *out of that very Context* — so installing it again to
measure it was a round trip to the device for a number in hand. Now taken directly, through the
same `reduced_potential` conversion (identical, not close), and read before any cross energy so it
is measured on the Context as propagation left it.

**This is exactly what Amber does.** `hamiltonian_exchange` sets `my_ene_temp%energy_1 =
my_pot_ene_tot` from the value the last dynamics step produced, swaps *coordinates* with
`mpi_sendrecv_replace`, and pays **one** `pme_force` call for the cross term its own source names
`my_pot_ene_tot_2   ! MY pot ene with THEIR coordinates`. The partner's half arrives by MPI. Cost:
one extra evaluation per replica per exchange, independent of ladder length.

**What I did NOT do, deliberately.** Rules can now declare which entries they read
(`required_entries`), honouring a promise `ExchangeContext` had always made — but the driver still
evaluates **every** entry. `rem_log.block_rows` indexes `u[state, state]` and `u[state, mate]`,
`exchange_free_energies` needs both cross terms of every proposed pair, and `exchange.csv` indexes
`u[state][walker]`: two conventions across three renderers, so a sparse matrix leaves holes some of
them index, and a hole reaches the reader as `nan` in `rem.log`. On a 2-state run sparsity wins
nothing anyway, and it only pays above ~5 states. The declaration is the rules' contract; the
density is the driver's business. Both facts are written into the code.

---

## Documented output names matched no file

`remd0.nc .. remdN.nc` and `remd<N>.cv.csv` appeared in CLAUDE.md, **all four** shipped
`configs/md/*.config`, `build/md.py`'s `state_trajectory` help, `storage.py`, `preflight.py`, two
published docs and two `example.in` files. **Nothing has ever written either name** — the search for
anything *constructing* one returns nothing, and `remd2.cv.csv` appears nowhere in the source.

The real names: `solute_state<i>_prod<N>.nc` and `whole_state<i>_prod<N>.nc`
(`state_trajectory_name` is the one authority), and `cv_state<i>.csv` with `cv_state<i>.json`
sidecars. `<stage>.cv.json` is the *stage* sidecar and was being quoted as though one name covered
both shapes.

Corrected everywhere the text is a live instruction. **Left alone** in `docs/history/**`,
`docs/release-notes/**` and `docs/reports/**`, which record what was true when written — the
convention your campaign journals already follow.

One test asserted `remd0.nc.1` does not exist: vacuously true of a name nothing writes, so it had
never tested anything. It now guards against a second **segment** (`..._prod2.nc`) appearing, which
is how an in-place `--extend` could actually break its contract. **That test is `@pytest.mark.slow`
and has not run yet** — see Pending.

---

## Post-hoc decomposition, where force groups cannot reach

`md_tools.openmm.decomposition` — an **import, not a command** (CLAUDE.md: do not add a fifth
command). It builds copies with charges or epsilons zeroed, so `EELEC`, `VDWAALS`, `1-4 EEL` and
`1-4 NB` come apart. Exact, not a fit: `U_nb(q, ε) = U_elec(q) + U_LJ(ε)` with disjoint parameters,
so zeroing ε leaves electrostatics *with its reciprocal sum and self-energy*, and zeroing q leaves
LJ *with its dispersion correction*. `decompose` checks the sum against the undecomposed total
rather than asserting it.

**Tested**: 6 tests, including one asserting the integrated System's serialisation and force groups
are unchanged.

---

## Test results

Full fast lane under the 0.5.3 checkout: **1697 passed, 537 deselected, 0 failed** (213.96 s).
New suites:

| file | tests | what it pins |
|---|---|---|
| `test_stage_output_sections.py` | 11 | census arithmetic, DOF, periodicity, method summary, single-sample guard |
| `test_exchange_entry_reuse.py` | 24 | diagonal reuse is the same arithmetic (exact, on a synthetic engine); `required_entries` matches what the sweep reads |
| `test_exchange_record_honesty.py` | 4 | `u_evaluated` derived, matrix stays dense |
| `test_energy_decomposition.py` | 6 | terms sum to the total; System unmutated |

**Engine-to-engine, 2 states, τ = (0, 0.1), same `prmtop`, 10 ns/state:**

| | Amber 26 REAF | MD-tools REST2 |
|---|---|---|
| acceptance | 0.47 / 0.49 | **0.514** |
| wall | 1338 s (646.9 ns/day) | 1173 s |
| free-NPT density | 0.9868 g/mL | 0.9876 g/mL |
| `rem.log` | 180207 bytes, 3006 lines | **byte-identical size and line count** |

Amber's own `pmemd.cuda.MPI` suite: 111 passed, 63 "possible FAILURE", **all** last-digit rounding
(worst 1e-1 absolute on a one-decimal column; the `.dif` files without a tolerance summary are
`dacdif` invoked without `-r`, not structural mismatches).

**8-rank smoke** (GPUs 1–8): ranks 0–7 mapped to distinct devices, all 7 neighbouring pairs
rendered with per-pair acceptance, `rem.log` 8 rows per exchange with the unpaired-replica
convention, 8 `cv_state<i>.csv` written with per-state digests in `restart.json`.

**Knowledge graph** refreshed: 7083 nodes / 12362 edges, `built_at_commit` moved to `7ed8a5a`.
Note `src/md_tools/build/` contributes **0 nodes** — `build` is one of 39 entries in graphify's
`_SKIP_DIRS`, so `build/md.py` and `build/record.py` are invisible to the graph. Not fixable from
configuration.

---

## Still pending

1. ~~The 8-state overnight run~~ — **DONE and verified.** See the section below.
2. ~~The slow lane~~ — **DONE.** 535 passed, 1 skipped, 40 m 43 s; the `_prod2` guard executed
   and passed. Its one failure was mine, and is classified in `d8f5185`.
3. ~~Interrupt/resume test of a ladder~~ — **DONE, and it found a real defect.** See below.
4. ~~The 8-state Amber comparison~~ — **DONE.** See below.
5. ~~The timing half of the three-combination matrix~~ — **DONE**, with its scope stated. See below.

Nothing on this list is outstanding. What remains is two judgement calls flagged at the end.

**Closed as infeasible:** reusing the post-exchange force call as the next segment's first step
(Amber's `runmd.F90:1718` trick). `openmm.Context` exposes only `getState`/`reinitialize`/`setState`
— no way to mark forces valid — and `step()` owns its evaluations. There is no OpenMM seam, and
`_install_owned` changes positions after every accepted swap anyway.

---

## The 8-state overnight run: completed and verified

τ = 0 → 0.5, eight states on GPUs 1–8 (eight identical RTX 3080s; GPU 0, the peer's A5000, left
alone). Ran the 0.5.3 checkout out of a throwaway venv, so it exercised these fixes rather than the
0.5.2 installed in the shared environment. Started 2026-09-14 21:07:11, `exit=0` at
2026-09-15 01:54:44 — **4 h 47 m**.

**Completion, from `restart.json`:**

```
run_status                   completed
steps_expected / completed   50000000 / 50000000
exchanges_committed          10000
production_ps_per_replica    100000.0          (100 ns per state, 800 ns aggregate)
whole_frames / solute_frames 1 / 50000
resumed_from_step            None
```

**All eight CV series re-hash to their recorded digests** — 50001 rows each (step 0 … 50,000,000 at
a 1000-step cadence, endpoints included exactly once, as the CV cadence invariant requires):

| state | τ | rows | digest | final step |
|---|---|---|---|---|
| 0 | 0.0 | 50001 | OK | 50,000,000 |
| 1 | 0.071429 | 50001 | OK | 50,000,000 |
| 2 | 0.142857 | 50001 | OK | 50,000,000 |
| 3 | 0.214286 | 50001 | OK | 50,000,000 |
| 4 | 0.285714 | 50001 | OK | 50,000,000 |
| 5 | 0.357143 | 50001 | OK | 50,000,000 |
| 6 | 0.428571 | 50001 | OK | 50,000,000 |
| 7 | 0.5 | 50001 | OK | 50,000,000 |

**Acceptance across all seven pairs** — even, with no impassable gap, which is what the per-pair
report exists to expose:

| pair | accepted/proposed | rate |
|---|---|---|
| 0 ↔ 1 | 3218/5000 | 0.644 |
| 1 ↔ 2 | 3127/5000 | 0.625 |
| 2 ↔ 3 | 3095/5000 | 0.619 |
| 3 ↔ 4 | 3099/5000 | 0.620 |
| 4 ↔ 5 | 3125/5000 | 0.625 |
| 5 ↔ 6 | 3137/5000 | 0.627 |
| 6 ↔ 7 | 3104/5000 | 0.621 |
| **overall** | **21905/35000** | **0.626** |

**Integrity:** `every_row_is_a_permutation: True` over 10 000 rows, no offenders;
`last_exchange: 9999`; `system_sha256 7521d7a0…` identical to the smoke runs; 22 solute atoms with
2 omega bonds excluded; final state→walker `[7, 2, 3, 1, 5, 4, 0, 6]`.

**The renderers agree with the arithmetic**, which is the point of keeping the matrix dense:
`exchange.csv` is **80001** lines = 10000 × 8 + 1, and `rem.log` is **90006** lines = 10000 ×
(8 + 1) + 6 header lines. The final `rem.log` block carries all eight states with correct pairing
and the unpaired-replica convention (`8  8  300.00 … 0.00`).

**The new `TIMINGS` block, with real numbers:**

```
# TIMINGS:
#   elapsed               17234.6 s
#   throughput            501.32 ns/day per replica, 4010.54 ns/day aggregate over 8 state(s)
#   per step              0.3447 ms
```

**On speed, stated carefully.** Amber's 2-state REAF reached 646.86 ns/day per replica on the same
system, against 501.32 here — Amber ~29% faster per replica. That is **not** a like-for-like engine
comparison: Amber ran 2 replicas on 2 GPUs with 1000 exchanges, this ran 8 on 8 with 10 000, so
exchange overhead, the N² cross-energy matrix and GPU contention all differ. The matched comparison
is still pending (item 4 above).

---

## Interrupt/resume found a real defect, now fixed

**A REST2 ladder left on the default output cadence could not be resumed after an interruption.**
Not an exotic configuration: `whole_output_interval` is unset by default in `REST2.config`, so a run
records `whole_output_interval_steps: None` and owes no whole-system frame at all — the 100 ns
eight-state run reported `whole frames: 1 (every None steps)` for exactly that reason.

The interrupt machinery was flawless. SIGTERM at an event boundary, checkpoint and run-state agreeing
on **step 266500 of 1000000**, exchange index **532**, four walkers, and — correctly — no
`restart.json`, because that is the *completion* manifest and an interrupted run has none. `--resume`
was then refused:

```
INVALID: 1 problem(s)
  - no whole-system frame was ever stored
```

`validate_replica_output` answers two questions: "is this complete and coherent"
(`expect_completed=True`, the extension path) and "may this be continued" (`expect_completed=False`,
the resume path, `driver.py:997`). The exchange-budget checks honour that flag; the coordinate-stream
check did not. So an interrupted run was judged against a finished run's property — one it had never
promised. The check three lines below already had the right shape, conditioning on whether the
*solute* interval was configured before demanding a solute frame; the whole-system check now matches
it. Not relaxed: an interval that *was* configured and produced nothing still fails, either way.

Fixed in `dcf9de0`, failing-test-first — three of five cases in
`tests/test_ladder_resume_without_whole_frames.py` failed on exactly this and pass after; the two
that must keep passing did so throughout; 178 tests across the five suites owning that validator
still pass.

**Confirmed end to end**, on the same interrupted ladder rather than only in unit tests:

```
run_status         completed
steps_completed    1000000 of 1000000
exchanges          2000
RESUMED FROM STEP  266500        <-- not None
cv series          4, rows [4001, 4001, 4001, 4001]
permutation rows   2000  ok=True
```

4001 = 1,000,000 ÷ 250 + 1 exactly, `system_sha256` unchanged across the interruption, permutation
integrity holding across the join.

It took three attempts, and the first two were my errors. Take one was sized so small (100,000 steps)
that it *finished* before the interrupt landed — testing instead the refusal of a re-run into a
populated directory, which named all 30 existing outputs on all four ranks, and is incidental
evidence for the "one identity per output directory" invariant. Take two cut correctly but I
restarted without `--resume`: CLAUDE.md's "`--resume` is not required" is scoped to a cMD **stage**,
while a **ladder** requires it, as `driver.py:2057` and `REST2/README.md:263` both say — the latter
adding that `--resume` needs no `restart.json`, which is exactly what an interruption leaves.

## The 8-state Amber comparison: Amber closes the ring, we don't

Matched — same `prmtop`, τ = 0 → 0.5 in 8 states, `gti_add_re=6`, `reaf_mask1=":1-3"`, NVT, 10 ps
exchanges, 1000 × 5000 steps = 10 ns/state, `exit=0` in 1404 s.

Parsed over all 1000 exchanges (8000 rows), Amber's `set_partners` pairs **1↔8** as well as the seven
adjacent pairs:

| pair | accepted/proposed | rate |
|---|---|---|
| 1↔2 | 502/1000 | 0.502 |
| 2↔3 | 508/1000 | 0.508 |
| 3↔4 | 492/1000 | 0.492 |
| 4↔5 | 496/1000 | 0.496 |
| 5↔6 | 524/1000 | 0.524 |
| 6↔7 | 540/1000 | 0.540 |
| 7↔8 | 534/1000 | 0.534 |
| **1↔8** | 272/1000 | **0.272** ← wrap-around |

Adjacent only: **0.5137**. The wrap-around is real, not a parsing artefact — all 858 rows with
|free energy| > 100 kcal/mol belong to the two pairs touching state 8, `(1,8)` 429 and `(7,8)` 429,
which is what pairing an endpoint across the full τ span produces.

**So 0.6259 (ours) and 0.5137 (Amber, adjacent) are not the same measurement**, and should not be
quoted side by side as an engine comparison. `alternating_pairs` does strict odd/even adjacent sweeps
with no wrap, as CLAUDE.md specifies; Amber spends an eighth of its attempts on a pairing worth 0.27.

Normalised per step, so the 10× length difference flatters neither side:

| | wall | per replica | per step | exchanges |
|---|---|---|---|---|
| Amber, 10 ns × 8 states | 1404 s | 613.6–618.5 ns/day | **~0.281 ms** | 1000 |
| MD-tools, 100 ns × 8 states | 17234.6 s | 501.3 ns/day | **~0.345 ms** | 10000 |

Amber ~23% faster per step, with a tenth of the exchanges to pay for — consistent with `AIS.md`'s
architectural finding that its λ change is a constant-block copy while ours re-uploads parameters.

**The gap between our own two runs is physics, not a defect.** Δτ per rung is 0.166667 at 4 states
against 0.071429 at 8 — 2.33× wider spacing over the same τ span — so acceptance falling from 0.6259
to 0.1917 is what a coarse ladder gives. The two runs corroborate each other.

## The three-combination timings, and what they do not measure

Identical fixed-τ cMD stages (τ = 0.25, 50 000 production steps), one GPU, one at a time:

| combination | wall | Speed (ns/day) |
|---|---|---|
| ff14SB + GBn2 | 5.42 s | **2175** ± 734 |
| ff14SB + TIP3P | 7.83 s | **1371** ± 463 |
| ff19SB + TIP3P | 8.10 s | **1215** ± 435 |

Implicit fastest; CMAP costs ff19SB ~11% against ff14SB on the same box. The expected ordering.

**What these do not measure: the switching cost.** τ is set once and never moves, so these are
*integration* rates. The re-upload that dominates an explicit-solvent AIS path — and the ~13× gain
`AIS.md` projects from carrying the dispersion tail analytically — is not exercised here at all. The
rms fluctuations are large relative to the means because a 50 000-step run's `Speed` column includes
startup, so treat these as indicative. A switching benchmark is the outstanding measurement for
`AIS.md`.

---

## What the reuse actually costs in precision

Measured on CUDA, physical GPU 1, the tau-scaled System, same Context and same instant:

| precision | noise floor (same read twice) | reuse vs evaluate |
|---|---|---|
| double | **0.000e+00** | **2-5e-11** |
| mixed  | 1.0e-05 - 3.4e-05 | 1.0e-04 - 7.0e-04 |

At double precision the two paths agree to 5e-11 reduced units, which is float64 rounding in the
`reduced_potential` arithmetic: the two computations ARE the same computation. At mixed precision --
what runs actually use -- neither side is bitwise stable: reading the same energy five times without
touching anything spreads by 3.4e-05, and re-installing the IDENTICAL configuration (which is what
the old path did, via save -> install -> restore) moves the energy by 3.5e-03, about a hundred times
the noise floor. The reuse difference sits below the perturbation the old path introduced, so the
reuse is not merely equivalent to it -- it avoids a disturbance the old path was causing.

This also explains why a ladder is not reproducible run to run: two identical runs gave different
equilibrated states (`eq_npt_free.xml` digests `647905a2...` and `d1454b50...`), because
mixed-precision CUDA is not bitwise deterministic and MD amplifies a 1e-5 difference exponentially.
Not a defect, but it means "the whole chain is reproducible" in `REST2.config` means same SEEDS, not
same bytes.

## Two judgement calls for you

Both are mine, made without you, and both could reasonably go the other way.

1. **Conditioning the whole-frame check on `whole_output_interval_steps`** is one fix; the other
   would be to have the resume path call a narrower validator instead of teaching this one to
   distinguish the two questions. I chose the former because it makes the two coordinate-stream
   checks symmetric and the asymmetry was the defect — but it does widen what
   `expect_completed=True` accepts, and that deserves a second opinion.
2. **`docs/release-notes/cuda-coverage-matrix.md` is deliberately not regenerated.** The invariant is
   satisfied by classifying the sites in the test module, which `d8f5185` does. The published
   document is a *record of a run*, and its own comment warns that an ordinary GPU run must not
   rewrite it as a side effect — so regenerating it from a lane I ran for another purpose would
   produce something that looks like evidence and isn't quite. Yours to trigger.
