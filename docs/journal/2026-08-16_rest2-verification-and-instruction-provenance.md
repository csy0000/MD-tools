# REST2 verified against its own energetics, and where the instructions live

**2026-08-16 (third pass).** Status: **REST2 confirmed working end to end**, by checking the
acceptance statistics against the Metropolis probabilities the code itself computed rather than by
reading an exit code. Also records where the refactor instructions live, and the one place the
stale-name audit legitimately matches.

## Is REST2 running fine? Yes — and here is what that rests on

An exit code of zero and `status: completed` were not sufficient to answer this. A REST2 run can
exit zero with a dead rung, a cut ladder, or every swap rejected for the wrong reason, so the
question was answered from the exchange log.

A fresh CPU run of the shipped smoke (3 rungs, s = 1.0 / 0.5625 / 0.25, T_eff 300 / 533 / 1200 K,
585-atom solvated box) recorded four attempts, all rejected:

| idx | pair | Δ (kJ/mol) | ln(acc) | P_acc | accepted |
|---|---|---|---|---|---|
| 0 | 0-1 | 23.59 | -9.46 | 7.8e-05 | 0 |
| 1 | 1-2 | 12.79 | -5.13 | 5.9e-03 | 0 |
| 2 | 0-1 | 9.25 | -3.71 | 2.5e-02 | 0 |
| 3 | 1-2 | 7.55 | -3.03 | 4.9e-02 | 0 |

Four attempts cannot distinguish "correctly rejecting improbable swaps" from "never accepting", so
the same ladder was run for twelve chunks:

    24 attempts, 1 accepted -> observed acceptance 0.042
    mean Metropolis probability over those attempts: 0.021
    highest single-attempt probability: 0.253

**Observed acceptance agrees with the probability the energies imply**, and the machinery does
accept when a favourable proposal arrives. That is the check that matters: the accept/reject step
is drawing against its own computed distribution rather than rejecting unconditionally.

Everything else the log shows is consistent with a healthy ladder: all energies finite (no NaN,
which is the failure that historically cut a ladder in two while the job still exited zero), both
neighbouring pairs attempted under an alternating phase, and Δ falling from 23.6 to 7.6 kJ/mol as
the replicas relax out of their shared starting state.

**Low acceptance here is the ladder, not a defect.** `smoke.yaml` spans 300 to 1200 K in three
rungs on a solvated box; energy variance scales with system size, so neighbouring distributions
barely overlap. The manifest declares `ladder_status: unvalidated` for exactly this reason -- it
exists to prove execution, not to sample anything.

**What this does NOT establish.** Nothing here says the ten-rung RGD ladder performs well. That
carries `ladder_status: pilot_supported` on separate evidence at 2 ns per replica, and it was not
run in this pass. An earlier smoke in this session showed chunk-level acceptances of 0.250 and
0.500, which look better than 0.042; that is small-sample noise on four attempts, not a regression
-- the twenty-four-attempt run is the one that agrees with its own predicted probability.

## Where the instructions live

The refactor was carried out against two instruction documents, both in this repository:

    claudecode-instructions/20260816_gpt_fix2.md            the seven-requirement first refactor
    claudecode-instructions/20260816_gpt_fix2_followup.md   run continuity and the public MD command

`20260816_gpt_fix1.md` is an earlier revision of the first, kept alongside them.

The two journals that describe the work -- `2026-08-16_first-refactor.md` and
`2026-08-16_run-continuity-and-md-cli.md` -- cite these by name, so keeping them in the repository
is what makes those citations checkable rather than references to something on one machine.

The follow-up's required reading names "CLAUDE.md"; there is no such file here, and it refers to
`20260816_gpt_fix2.md`.

## The one legitimate stale-name match

The instruction files necessarily spell the retired identity: their content is the directive to
remove it. Run literally, the acceptance check therefore reports matches:

    14 matches, ALL of them inside claudecode-instructions/, and none anywhere else

For routine use, exclude that directory and the check returns zero:

    ... -g '!claudecode-instructions/**'      -> 0 matches

This is the intentional exception the follow-up asks to be explained rather than hidden. The
alternative -- deleting or ignoring the instructions -- would remove the record of what the
refactor was asked to do, which is worse than one documented exception. The same reasoning applies
to the journals, which is why they describe the audit pattern instead of quoting it: a file that
matches the audit merely by discussing it turns a clean gate into a permanent non-zero number, and
a gate that always reports the same count stops telling anyone anything.
