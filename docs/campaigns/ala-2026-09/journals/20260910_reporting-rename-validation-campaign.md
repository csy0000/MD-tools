# 2026-09-10 — validating the reporting rename with a 0.01× ALA campaign

MD-tools renamed its reporting parameters and the files they produce. This records the campaign
that validated the change end to end, and the defects it found that the test suite did not.

MD-tools `7d76b05` and `b5783bd` on branch `dev`.

## What changed upstream

| old | new |
|---|---|
| `solute_printout` | `crd_printout_solute` |
| `system_printout` | `crd_printout_whole` |
| — | `info_printout` (the state table, previously unnamed) |
| `<stage>.dcd` | `solute_prod<N>.nc` and `whole_prod<N>.nc` |
| `<stage>.csv` | `mdout.csv` (production), `mdout_<stage>.csv` (every other stage) |
| `remd<i>.nc` | `solute_state<i>_prod<N>.nc`, `whole_state<i>_prod<N>.nc` |
| — | `exchange.csv` |

`state<i>`, not `rep<i>`, is deliberate: MD-tools' per-state files follow the Hamiltonian.
Amber and GROMACS write per-WALKER trajectories, which is why `remdtrajtemp` and cpptraj's
temperature sort exist at all. The name now says which convention a file follows.

## The campaign

Six runs at one hundredth of the production budget — 5 ns/state REST2 (4 rungs implicit,
6 explicit), 5 ns hot cMD at τ = 0.5, 10 ns cold cMD at τ = 0 — in both solvents, packed onto
nine GPUs. It lives at `data/ala-campaign-0p01x/` (untracked: `data/` is generated output).

**This validates mechanism, not physics.** 5 ns supports no convergence or sampling claim and
none is made. What it proves is that every protocol writes the streams its configuration asks
for, under the right names, with the frame counts the intervals imply, and that a ladder's
per-state files record which state they hold.

`verify_layout.py` in that directory is the check, and it passes on all six cells:

| cell | solute stream | whole stream | state table |
|---|---|---|---|
| cMD implicit | `solute_prod1.nc`, 22 atoms | `whole_prod1.nc`, 22 atoms | `mdout.csv` |
| cMD explicit | `solute_prod1.nc`, **22 atoms** | `whole_prod1.nc`, **1796 atoms** | `mdout.csv` |
| REST2 implicit | `solute_state{0..3}_prod1.nc` | `whole_state{0..3}_prod1.nc` | `exchange.csv`, `rem.log` |
| REST2 explicit | `solute_state{0..5}_prod1.nc` | `whole_state{0..5}_prod1.nc` | `exchange.csv`, `rem.log` |

The explicit row is the point of the whole exercise: a 22-atom solute stream beside a 1796-atom
whole stream, from one run, at different cadences.

## Four defects the suite did not catch

1. **Every explicit-solvent stage died at its first reported frame.** The stage reporter passed
   the box as three diagonal scalars where the writer takes the three box vectors. No test
   caught it because every stage test in the suite is implicit, where `periodic` is False.
2. **A ladder wrote its whole-system stream at the state table's cadence** — ten times the
   frames configured, silently. The suite *did* have a test for this; it was failing on the old
   key names, which hid the real defect. A test that is red for its own reason cannot catch
   anything.
3. **A crashed chain could not be restarted at all.** `md-run` ran the output-collision check
   before the stage could consult its completion record, so a completed stage was never skipped
   — while the refusal advised `--resume`, which the cMD path rejects by name.
4. **`--overwrite` governed neither per-state trajectory nor the phase-space sidecar**, because
   the inventory named files no run writes any more.

## The pattern worth remembering

`crd_printout_whole` defaults to **0**. Everything that consumes the whole stream — an AIS
source ensemble, a ladder's CV frame naming, progress reconciliation — silently gets nothing
unless a configuration asks for it. Seven test fixtures and the shipped AIS config all set
`info_printout` where they meant `crd_printout_whole`, and every one of them looked like a
different failure: a missing source, an alignment error, a run that produced no frames. When
something downstream of the whole stream behaves oddly, check that interval first.

## Not addressed

- A `build-md` cMD tree writes no `resolved_run.yaml`, so AIS cannot source from one at all.
  This predates the rename — only the `md-openmm setup` route writes that record — but it means
  the corrected AIS default still does not work end to end from a `build-md` tree.
- A REST2 ladder writes no `mdout.csv`. `info_printout` now feeds nothing on that path; the
  ladder's state data lives in `REST2.nc` and `rem.log`. Not a regression, but the key is
  accepted and silently ignored there, which is what the strict schema exists to prevent.
- `REST2.solute.nc` still duplicates the per-state solute streams, and a ladder's CV series is
  still `remd{i}.cv.csv` — the one place the old name survives.
