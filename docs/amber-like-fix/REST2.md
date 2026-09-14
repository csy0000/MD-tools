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
| `test_exchange_entry_reuse.py` | 24 | diagonal reuse is bit-identical; `required_entries` matches what the sweep reads |
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

1. **The 8-state overnight run** — τ = 0 → 0.5, 100 ns/state, 800 ns aggregate, GPUs 1–8, started
   21:07:11. At 21:52 it was ~18.5% in (9250 CV rows of ~50000), projecting a finish near **01:10**.
   Outputs: `<scratchpad>/reaf/md_script8/`. **Verify `restart.json` completion, the 8 CV series
   against their recorded digests, `rem.log`/`exchange.csv` at N=8, and checkpoint integrity.**
2. **The slow lane** — my `_prod2` guard has never executed.
3. **Interrupt/resume test of a ladder** — the clean run does not exercise it.
4. **The 8-state Amber comparison** at matched settings; only 2 states were compared.
5. **The timing half of the three-combination matrix.** The structural findings above are
   platform-independent, but ns/day and the switch fraction for ff14SB+GBn2, ff14SB+TIP3P and
   ff19SB+TIP3P all need GPUs. See `AIS.md` for why the three are expected to differ — GBn2 is
   already on the fast switching path, and only ff19SB carries the CMAP residue (16 maps /
   1 torsion, against 22 particles + 98 exceptions + 29 torsions that become offsets).

Items 2–4 need GPUs, which the overnight run holds.

**Closed as infeasible:** reusing the post-exchange force call as the next segment's first step
(Amber's `runmd.F90:1718` trick). `openmm.Context` exposes only `getState`/`reinitialize`/`setState`
— no way to mark forces valid — and `step()` owns its evaluations. There is no OpenMM seam, and
`_install_owned` changes positions after every accepted swap anyway.
