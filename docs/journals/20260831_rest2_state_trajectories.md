# REST2 fixed-state trajectories, CMAP duplication, REM log — MD-templates record

**Date** 2026-08-31
**Branch** `feat/rest2-state-trajectories`, from `b919623da3f25e2a984380b653bbb7600263b0fb`
**Instruction** `claudecode-instructions/20260831_rest2-state-trajectories-cmap-rem-log.md`

**This milestone is not complete.** Five modules landed; the integration that replaces the
coordinate layout has not started. Nothing below claims a check that did not run.

---

## What landed

| commit | § | change |
|---|---|---|
| `470cd4d` | 7 | `rem_log.py` — the Amber26 H-REMD projection |
| `837deda` | 1 | tau as the only public/persisted coordinate; `rest2-no-bond-angle-omega/v1` |
| `1045729` | 2 | CMAP duplication and torsion re-routing |
| `63cdd7f` | 3 | the omega exclusion recorded as exact torsion indices |
| `e487d8f` | 6 | `amber_trajectory.py` — genuine Amber NetCDF, one file per state |

Full suite at every commit: **808 passed**, 0 failed, 0 skipped (~6m45s). Ruff delta zero on the
changed files; no added line exceeds 99 characters. 38 new tests.

---

## §1 — tau is the only coordinate

The Hamiltonian had two names for one thing: `tau`, and a derived `s = (1-tau)^2` that was printed,
persisted in YAML and manifests, and exported as `scale_factor_for_tau`. A derived quantity that is
also stored is a second thing to keep consistent, and the one that drifts is never the one anybody
checks.

`scaling_for_tau(tau)` returns both factors together and they are threaded down to the force
handlers. The `math.sqrt` that used to be recomputed inside `_scale_nonbonded` is gone: one square
root in three places is three chances for one of them to be changed alone.

Removed from public and persisted surfaces: `scale_factor_for_tau`, `derived_scale_factor_s`,
`derived_scale_factors_s`, `scale_factor_s`, `derived_s`, `derived_sqrt_s`,
`derived_solute_environment_coupling_sqrt_s`, and the `s = …` text in CLI output and generated
protocol headers.

`REST2_IMPLEMENTATION` names the convention and records what is scaled and what is deliberately left
alone. It participates in identity: two runs agree only if they agree about what REST2 meant.

**AIS is a deliberate exception.** Its switching observations carry `s` and `sqrt_s` columns, and
datasets written with them exist on disk. That schema is AIS's, not REST2's; renaming it would break
readers of data this repository already produced. It now takes its value from the one REST2 entry
point through a helper that says so.

### Two renames that would have changed the physics

Both were mechanical, both were caught by the suite, both are reverted and now stated at the point
of use:

- the generalised-Born energy scales by the solute-solute factor **in full**, deliberately, because
  its non-polar term has no charge dependence and charge scaling would leave it untouched;
- a nonbonded exception with exactly one solute partner follows the **solute-environment** factor,
  as an unexcepted solute-environment pair does.

---

## §2 — CMAP duplication, and a defect found in passing

A CMAP map is a shared lookup table referenced by every torsion of the same residue type, so "is
this map the solute's" has three answers, not two.

The previous code scaled only maps used **exclusively** by solute torsions and skipped shared ones.
That looks conservative and is the opposite error: a solute torsion sharing a map with a non-solute
one received **no scaling at all**, so the Hamiltonian silently stopped being the one the ladder
claimed.

Each shared map now gets exactly one duplicate; the duplicate carries `(1-tau)^2`; only wholly-solute
torsions are redirected to it; the original stays byte-identical for the environment. One copy per
MAP, not per torsion. At tau = 0 the cold rung gains nothing.

Live switching needed care. `TauSwitcher` derives the duplicate→original mapping once from the
untouched base, in the same deterministic order the copies are appended, so a duplicate is matched
back without a side table that could drift from the System. `_restore_cmap` restores a duplicate
from the **original** it was copied from — without that it read past the end of the reference and
the switcher compounded scaling instead of reapplying it. A test switches 0.5 → 0.25 → 0.5 and
requires a constant map count and each tau applied to the unscaled energies.

---

## §3 — the omega exclusion, recorded as torsions

The exclusion was persisted as a pair of atom indices, which needs a force field to mean anything.
`torsion_exclusion_report` records every torsion index each excluded bond left unscaled, the count
of scaled solute torsions, and `OMEGA_DETECTOR_VERSION`, against the same System the ladder was
built from. It applies the same predicate `_scale_torsions` does, so the record cannot describe a
different exclusion than the one performed — asserted by scaling a System and checking the protected
torsions kept their force constants while the others took `(1-tau)^2`.

The two-route classifier itself is unchanged; this records what it decided.

---

## §6 — Amber NetCDF trajectories

One file per fixed thermodynamic state. A walker moves between states on an accepted exchange, so
the trajectory of a state is not the trajectory of a walker: after a swap the next frame in
`remd2.nc` comes from whichever configuration now occupies state 2.

The filename carries the state index and never tau. `state_index_from_name` exists only to validate
a group file that names its own outputs; nothing sorts trajectories lexicographically or parses a
number out of a path to decide order.

`validate_state_outputs` refuses duplicate, missing, non-contiguous or out-of-order outputs before
any file is created — position in the group file *is* the state index.

Coordinates convert from nanometre to angstrom once, at this boundary. NVT only: the cell is written
per frame from a fixed box, and there is no variable-cell handling.

---

## §7 — the H-REMD log

`rem.log` is a projection of committed exchange rows, not an authority, and is rewritten in full and
replaced atomically so a crash cannot leave a torn log disagreeing with `exchange.nc`.

Pairs and outcomes are read from the committed proposal and acceptance matrices rather than
re-derived from the exchange rule, which could have been reconfigured since the rows were written.

### Grammar, and two points that are ours

The Amber26 manual is **not installed** and the cpptraj parser source is **not shipped** (header and
object only). AmberTools26 does ship genuine Amber-produced H-REMD logs and the parser binary, which
together are better ground truth than prose. Every field width comes from `test/h_rem/rem.log.save`:

```
Rep# %6d  Neibr# %6d  Temp0 %10.2f  PotE(x_1) %10.2f  PotE(x_2) %10.2f
left_fe %10.2f  right_fe %10.2f  Success %5s  rate %12.2f          (79 chars)
```

- **Unpaired-replica sentinel.** Amber always pairs every replica using a wrap-around. This
  repository pairs adjacent states without wrapping, leaving two unpaired on alternate attempts.
  `Neibr# = Rep#` parses cleanly, and is ours, not Amber's.
- **`Success rate`.** Amber's values could not be reproduced from any documented reading — it reports
  `2.00` at the first exchange and the two rows of one pair disagree. Its influence was measured
  instead: replacing the whole column with `-999.99` leaves cpptraj's reconstruction identical, so it
  is not load-bearing. A plainly defined cumulative pair acceptance is written and is **not** claimed
  to reproduce Amber's.

---

## Parser evidence

Every claim below is a command that ran, not an inspection:

| check | result |
|---|---|
| cpptraj reads our log | *"6 Hamiltonian reps … should contain 6 exchanges"*, no errors |
| `runanalysis remlog … crdidx` | reconstructs our history: after one accepted 0↔1 swap, `2,1,3,4,5,6` |
| Amber's own log re-rendered | every row byte-identical — caught the block header a column too wide |
| cpptraj reads our trajectory | *"a NetCDF (NetCDF3) AMBER trajectory with coordinates, time, box"* |
| unit conversion | 0.0/0.1/0.2 nm read back through cpptraj as 0.000/1.000/2.000 Å |
| parent + extension concatenation | six frames, times `[2,4,6,8,10,12]` — no duplicated, no missing boundary frame |

Environment: cpptraj **V7.6.2** (AmberTools26) with sander and tleap; openmm 8.6.0; netCDF4 1.7.4;
mpi4py 4.1.2; 9 GPUs (device 0 RTX A5000, 1–8 RTX 3080), driver 580.173.02.

**Local validation only.** No CI workflow ran on these commits and none is claimed.

---

## Unfinished, in dependency order

Each item names what it blocks, so the next session can start without re-deriving the order.

1. **§5/§6 — wire `amber_trajectory` into the driver.** Replace
   `positions[frame, walker, atom, spatial]` with N per-state writers. At a frame event, for each
   state write `configurations[state_to_walker[state]]`. Event order stays
   `propagate → exchange → full frames → solute frames → checkpoint`, so a coincident frame is
   post-exchange. *Blocks 2, 3, 4, 6.*
2. **§6 — version the `exchange.nc` schema** and add an explicit read-only refusal for older
   bundled-coordinate files. Migration is not appropriate here: the layouts hold different things.
3. **§9 — the multi-file commit protocol.** Write every state trajectory, sync all, advance one
   global committed-frame marker, sync the exchange record. On continuation, validate equal
   committed times/steps and state identities across all trajectories; a short or missing state file
   is corruption and must be refused, never padded.
4. **§7 — the `--rem rem.log` flag**, regenerating the log from committed rows on every write and on
   restart. `--verify-only` compares it and stays read-only.
5. **§4 — resolved YAML per-state records**: index, trajectory basename, exact tau, and
   `effective_temperature_k = physical / (1 - tau)^2` for reporting only.
6. **§8 — neighbouring-pair-only completion report**, cumulative from committed rows; reservoir
   attempts reported separately for rREST2.
7. **§3 — ambiguous-candidate evidence** and the six residue cases (ACE-ALA-NME, X-PRO, N-methyl
   amide, ligand amide, macrocyclic amide, unrecognised modified residue).
8. **§14 — the extension-directory interface.** `--extend` appends in place today, which §14
   forbids for this example. It needs a mode that consumes the parent's terminal checkpoint and
   writes a new directory, leaving the parent byte-for-byte unchanged. *Blocks the 5 ns validation.*
9. **§11/§12 — MD-project pin, workflow and provenance; user documentation.**
10. **§14 — the 5 ns + 5 ns validation itself**, once 1–8 exist. Two 6-state explicit-solvent runs,
    roughly 60 ns of aggregate MD.

The concatenation requirement in item 10 is already established at the writer level (above), so the
5 ns run has to demonstrate it end to end rather than discover it.

---

# Second pass — §15 superseding correction

**MD-templates** `feat/rest2-state-trajectories` → `c50a13e1ac7dbb8b2420f7b08d507306777fa0e2`, from `caad06c`.
Suite **822 passed**, 0 failed, 0 skipped at every commit. Ruff delta zero; no added line over 99.

## The generalised-Born contribution now scales by `1 - tau`

§15 supersedes §1's rule and the decision recorded in the first pass of this journal. The whole GB
contribution is the solute's coupling to a continuum standing in for solvent, so it is a
solute-environment interaction and follows the linear factor.

Every term is still multiplied by one global parameter rather than through charges: GBn2's
non-polar term has no charge dependence, so charge scaling would silently leave it at full strength.
A test asserts exactly that on a charge-free term — `7.0` becomes `5.25` at tau = 0.25.

## The identity is v2, and v1 is refused for continuation

```yaml
rest2_implementation:
  name: rest2-no-bond-angle-omega
  version: 2
  generalized_born_scale: "1-tau"
```

`require_compatible_implementation` refuses to continue a run recorded under v1, with the reason
attached: a version bump here says the energy function changed, so continuing across one would join
samples from two different ensembles. v1 is kept as a historical identity. No old manifest or
journal was rewritten.

## Energy tests, not parameter tests

The ordinary nonbonded force and the GB force are placed in separate OpenMM force groups and
evaluated independently, so each ratio is measured on the energy that force actually produces:

| tau | GB ratio | nonbonded ratio |
|---|---|---|
| 0.0 | 1.0 | 1.0 |
| 0.25 | 0.75 | 0.5625 |
| 0.5 | **0.5** | **0.25** |

At tau = 0.5 GB halves while nonbonded quarters. That divergence is the correction, and a test that
only inspected parameters would not have seen it. Cloned states and live switching are both covered,
and repeated switching 0.5 → 0.25 → 0.5 returns to the same energy rather than compounding.

`test_fixed_tau_md.py` is updated where it pinned the superseded rule.

## Second pass — everything remaining, and four defects the tests could not see

### The class of defect, and the fix for the class

Every test of `replica_driver` reads its source text or binds one of its methods to a stub.
Nothing executed `_loop`. Four things were committed and pushed in that state, each of which a
single real run breaks on:

| defect | consequence | committed in |
|---|---|---|
| `self.trajectories` used by the frame event and at close, **never assigned anywhere** | the per-state trajectories were never created; a real run raised `AttributeError` at its first frame | `1614eb5` |
| `completion_report` called without being imported | `NameError` at completion | this pass |
| `--rem` parsed, validated and documented, never put into the driver's `files` | `_write_rem_log` returned early every time; no real run wrote a REM log | `0cf93f6` |
| `rem_log`, `amber_trajectory`, `state_trajectories` missing from the generated-project copy list | every generated ladder raised `ImportError` at launch | `1614eb5` |

The third was found by `ruff --select F821`; the first and fourth by writing the test that should
have existed all along.

`tests/test_grouped_run_end_to_end.py` now runs one complete grouped REST2 ladder — two states,
alanine dipeptide in vacuum, CPU platform, a few seconds — through the real `openmm_md.py` entry
point with a real group file, and asserts the state trajectories, the REM log, the completion
report and a continuation. `tests/test_generated_project_completeness.py` checks the copy list
against the driver's actual import-time closure, computed from the source rather than remembered.

### §4 — the resolved state records

`restart.json` gains `states`: index, trajectory basename, exact tau, and
`effective_temperature_k = T_physical / (1 - tau)^2`, derived and reporting-only.

Not in `protocol.describe()`, which feeds the identity a continuation compares key by key: a
reporting field there would make every previously written file report a difference and refuse to
continue. A test states exactly that, because the mistake is an easy one and looks tidier.

### §8 — the completion report

Neighbouring pairs, an overall figure, and nothing else, from one builder that renders both the
persisted record and the terminal summary so the two cannot drift.

Round-trip counting was **removed** rather than merely unprinted, together with
`count_round_trips` and `round_trip_report`. It needs burn-in and window choices this layer has no
basis to make, and a number printed at completion is quoted as though the choice had been
justified. `mapping_is_permutation_every_iteration` stayed and is recorded as `mapping_integrity`
— that is storage integrity, not analysis.

### §3 — the omega record

`solute.yaml` now carries the detection route, the detector version, the ambiguous candidates with
evidence (bond, both residues by name and index, and the classifier's own sentence), and the exact
`PeriodicTorsionForce` indices each excluded central bond protects.

`tests/test_omega_classification_cases.py` states the six cases, each with its reason. The two
worth repeating: an **N-methyl amide is ordinary**, because an N-methylated amide isomerises more
readily than an N-H amide rather than less; and the **≤ 7-membered ring bound** is what makes the
ligand route correct for macrocycles, since every backbone nitrogen of a cyclic peptide is "in a
ring" and an unbounded `[NX3;R]` test would free every macrocyclic omega for scaling.

### §14 — `--extend-from`

The parent is opened read-only and every refusal is decided before anything local exists.
`AmberTrajectoryWriter.open_existing` reopens a state trajectory for append at the committed-frame
marker, checking state index, tau, atom count and conventions first;
`StateTrajectorySet.continue_from` inspects the whole set read-only and refuses before touching
any file, so a corrupt member cannot be found after the healthy ones were appended to.

Two things worth recording:

* The first version assumed the parent's files were named `exchange.nc` and `checkpoint.nc` — the
  names §14's illustration uses and that nothing real uses. A generated ladder writes `rest2.nc`
  and `rest2_checkpoint.nc`. Only `restart.json` is assumed now; the rest is read out of it.
* **The continuation is proved, not asserted.** The out-of-place extension runs beside an in-place
  `--extend` of the same parent and the frames must be *identical*. A rerun from the input
  coordinates, a coordinate-only restart or a dropped RNG state each break that at once.

**rem.log across the boundary** is segment-local, from 1, as Amber does on restart, with
`numexchg` counting that segment's blocks — which cpptraj checks. Both conventions were measured
against the installed cpptraj (V7.6.2) first: it parses either and never reads the block number,
so this is a choice and it is recorded as one. The absolute offset lives in the extension
provenance as `first_exchange_number`.

### The staging exception

`resolve_output_root` still refuses an output inside a git working tree — an ignore rule is one
`git add -f` from being wrong. One exception now exists for the staging step the registration
pipeline needs, and it is checked rather than declared: `MD_TEMPLATES_ALLOW_STAGING=1`, a
`.md-staging` marker visible in the tree, and `git check-ignore`'s opinion about a path *beneath*
the root — not the root, which is usually tracked. Any one missing and the refusal stands.

### Commands and results

```bash
export PATH=/path/to/software/md-stack/envs/openmm-rest2/bin:$PATH
python -m pytest tests -q -p no:randomly        # 900 passed
ruff check --select F821 src tests              # all checks passed
```

Without the scientific environment on `PATH`, `test_a_stage_runs_with_md_templates_unavailable`
fails: the generated launcher resolves `python` from `PATH` and finds an interpreter with no
OpenMM. That is the test working.

## Still unfinished in this repository

Nothing from the dependency list. Items 1–8 are all complete, and the ALA 5 ns + 5 ns validation
is recorded in the MD-project journal, `docs/journals/20260831_rest2-state-trajectories.md`, which
is where the dataset and the registration pipeline live.

**Local validation only.** No CI ran on these commits and none is claimed.
