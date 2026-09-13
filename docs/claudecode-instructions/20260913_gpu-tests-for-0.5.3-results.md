# GPU tests for 0.5.3 — results

Answers `20260913_gpu-tests-for-0.5.3.md`. Run on **GPUs 7 and 8** (4 rungs oversubscribed;
`select_device_for_rank` round-robins when ranks exceed devices), alanine dipeptide in GBn2, 2 fs,
300 K, seed 20260913, from branch `gpu/0.5.3-evidence`:

```bash
CUDA_VISIBLE_DEVICES=7,8 pytest tests/test_hprest2_gpu_evidence.py -q -p no:randomly -s
# 4 passed in 49.89s
```

Every number below is from one run of that file; the tests write them to `$HPREST2_GPU_REPORT`
rather than only asserting.

## Before the numbers: the fix was incomplete, and nothing on CPU could have shown it

`b9173b7` taught the group-file PARSER that a line with no `-c` is legal when `--extend-from` is in
force. Both of its consumers still read the key unconditionally, so **an extension still aborted**:

| where | what happened |
|---|---|
| `executor._announce` | `KeyError: 'coordinates'` on every rank, immediately after the header was printed — so the run looked as though it had started |
| `executor.run_grouped` | the same error one step later, building the driver's `files`, visible only once the first was fixed |

Found by extending a real two-rung ladder **on the CPU**, which is the first thing that gets past
the parser at all. Both are fixed in `9323f1a`, with two tests: one drives `_announce` with the
exact shape `--extend-from` emits, and one refuses any unguarded `[...]["coordinates"]` anywhere in
the executor — this was one omission in two places, and a third would look the same.

**`md-openmm md-run` defines neither `--extend` nor `--extend-from`.** Its parser stops at
`--resume`/`--overwrite`; the generated `REST2.py` carries both and forwards them to the executor.
An extension is therefore launched as

```bash
mpirun -n 4 python REST2.py -p ../built.pdb -s ../built.xml -ng 4 --extend 10 --extend-from ../baseline
```

## 1. REST2 still runs and still exchanges

200 exchanges × 250 steps = 50,000 steps per state. (It was 40 exchanges first; 60 proposals put
an uncertainty of about 6 percentage points on every rate below, which is too loose to compare with
anything, so the ladder was lengthened to 300 proposals for two more minutes of GPU time.)

| measured | value |
|---|---|
| `run_status` | `completed` |
| `exchanges_committed` | 200 of 200 asked for |
| `steps_completed` | 50,000 |
| `mapping_integrity` | 200 rows, `every_row_is_a_permutation: true`, no offending rows |
| `rem.log` | 200 exchange blocks, one row per rung, partner relation symmetric in every block |
| acceptance, overall | 154 / 300 = **51.3 %** |
| pair 0↔1 (τ 0 → 0.1667) | 45 / 100 = 45 % |
| pair 1↔2 (τ 0.1667 → 0.3333) | 53 / 100 = 53 % |
| pair 2↔3 (τ 0.3333 → 0.5) | 56 / 100 = 56 % |
| final state→walker | `[2, 3, 0, 1]` |

## 2. Extension across a boundary

Launched with **no `-c`** — the case that used to abort every rank.

| measured | value |
|---|---|
| parent `steps_completed` | 50,000 |
| child `resumed_from_step` | **50,000** (exactly, not approximately) |
| child `steps_completed` | **52,500** = 50,000 + 10 × 250 |
| parent afterwards | byte-identical; **0** files changed of the full digest set |
| parent → child final map | `[2, 3, 0, 1]` → `[0, 2, 3, 1]`, a permutation |

**The trajectory joins rather than restarting, matched by WALKER.** A per-state file changes
occupant at every accepted exchange, so comparing state *s* in the parent with state *s* in the
child measures the exchange, not the join. Frames are written with the state→walker mapping in
force, so the mapping says where each walker was: the parent's final row and the child's first.
RMS distance, per walker:

| walker | parent state → child state | matched join (Å) | same-state join (Å) | one interval in parent (Å) | parent first→last (Å) |
|---|---|---|---|---|---|
| 0 | 2 → 3 | **1.39** | 4.02 | 4.99 | 4.08 |
| 1 | 3 → 2 | **2.15** | 4.34 | 1.91 | 2.37 |
| 2 | 0 → 0 | **1.03** | 1.03 | 0.97 | 4.56 |
| 3 | 1 → 1 | **1.93** | 1.93 | 4.77 | 2.18 |

The two walkers that changed state across the boundary show it plainly: matched 1.39 Å and 2.15 Å
against same-state 4.02 Å and 4.34 Å. Every matched join is one exchange interval of ordinary
motion, below the parent's own first-to-last spread; a re-thermalised restart would not be. The
first version of this table was per state and therefore a range check — the numbers above are the
corrected measurement, not a re-tuned threshold.

### `-c` supplied to an extension is inert

Same extension run twice, once with `-c eq.xml`:

| | without `-c` | with `-c` |
|---|---|---|
| `resumed_from_step` | 10,000 | 10,000 |
| `steps_completed` | 10,500 | 10,500 |
| final state→walker | `[1, 3, 2, 0]` | `[1, 3, 2, 0]` |
| `rem.log` decisions | identical, exchange for exchange | |

## 3. REST2 with harmonic restraints

**This did not exist and had to be implemented.** `umbrella.file` was accepted only with
`protocol: umbrella`, refused by name for everything else, and nothing in `remd/` had ever seen a
restraint; there is no `nmropt`/DISANG surface anywhere — restraints are declared in this package's
own `umbrella.yaml` against named collective variables. `a251295` makes `umbrella.file` legal with
`REST2`/`rREST2`, meaning **the same restraints on every rung**, resolved by the preflight against
`collective_variables.file` and added inside `build_rung_systems` **after** each rung is scaled.

Run: 4 rungs, 20 exchanges × 250 steps, harmonic on φ (`phi_ALA`, atoms 4-6-8-14) at −60°,
500 kJ/mol/rad².

| measured | value |
|---|---|
| `run_status` | `completed`, 20 exchanges |
| recorded restraint | `scientific_identity.torsion_restraints`: definition, cv, form, centre, constant, atom indices, `scaled_by_tau: false` |
| φ, propagated frames (80) | mean \|Δ\| from centre **3.61°**, max **13.54°** |
| φ, the 4 excluded step-0 rows | 100.97° — the shared, unrestrained starting structure, one per state |
| `log alpha` with bias − without, on this run's own states | **−3.97 × 10⁻⁴ kJ/mol** (tolerance 0.05, CUDA mixed precision) |
| the same identity, exactly | ≤ 1 × 10⁻⁶ kJ/mol on CPU, `tests/test_ladder_torsion_restraints.py` |

### The second bullet of your test 3 cannot hold, and this is why

You asked that a restrained and an unrestrained ladder, from the same seed and start, accept the
**same exchanges**. They will not, and expecting it would hide the real property. An identical
restraint cancels from `log alpha` **evaluated at the same configurations** — that is exact, and it
is what the two numbers above measure — but it also changes the FORCES, so the two ladders'
trajectories diverge from the first propagation and their later decisions differ for an honest
reason.

What is testable exactly is the identity, and it is tested twice: at fixed configurations on the
CPU (1e-6 kJ/mol, with a counter-example where one rung is restrained differently and the
difference is **not** zero), and on the states this CUDA ladder actually visited.

## What was left open, and what became of it

* **rREST2 + reservoir + restraints — now REFUSED, not merely cautioned.** A reservoir sample is
  drawn from a distribution generated WITHOUT the bias, so refreshing the top rung would install an
  unrestrained configuration into a restrained ladder, which then samples neither ensemble with
  nothing in the output saying so. `umbrella.file` together with `reservoir.enabled` is refused by
  name at configuration time, with a test for both directions; the rREST2 README says the same.
* **Acceptance statistics — tightened.** 300 proposals instead of 60.
* **Trajectory continuity — sharpened.** Walker-matched, as above, instead of a per-state range
  check.
* **A restrained ladder still cannot be exported**, and the reason is not the one I first gave.
  `verify_rungs.py` rebuilding rung *i* by scaling rung 0 is a real obstacle — the restraint is
  added after scaling — but it is second. The first is that a restrained ladder must declare
  `collective_variables.file`, and `_check_collective_variables` refuses a file with
  `interval_steps: 0`, so such a ladder always REPORTS a CV series, which a bundle does not carry
  or reproduce. Making export reachable therefore needs a deliberate schema decision — allowing a
  CV definition that exists only to resolve restraints, reporting disabled — before any exporter
  work is worth doing. I started the exporter plumbing, found that it could never be exercised, and
  reverted it rather than leave a `settings.json` field that is always empty.
* **Restraint strength vs acceptance** was not studied.

## Two defects in my own tests, for the record

Both were mine, not the engine's, and both were fixed rather than loosened:

* I misparsed `rem.log`. The exchange number is on a `# exchange N` comment line, and the data rows
  are `Rep# Neibr# Temp0 …`; reading three integers off a data row finds `300.00` and silently
  drops every row. The log was well formed; the test saw nothing in it.
* I judged the restraint by its worst frame, which measured the startup transient — the shared
  unrestrained start at φ = −161° — rather than the hold. It now excludes each state's step-0
  observation, which the series marks with `exchange_attempt = -1`.

## Not done, as asked

Nothing was merged to `main`. The work is three commits on `gpu/0.5.3-evidence`, on top of
`b9173b7`: `a251295` (restraints on a ladder), `9323f1a` (the two executor fixes), and the test and
documentation commit carrying this file.
