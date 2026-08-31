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
