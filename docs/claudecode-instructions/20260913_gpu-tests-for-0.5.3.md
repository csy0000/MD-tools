# GPU tests for 0.5.3 — REST2, extension, and REST2 with harmonic restraints

**For the agent working on MD-tools.** Branch `fix/hprest2-findings` carries three fixes and their
unit tests. Those tests are CPU-only and prove the parser and the naming rule. They do **not**
prove a ladder still runs, still exchanges, and still continues correctly across a boundary on a
GPU, which is the thing that actually broke. This asks for that.

**Two GPUs are reserved for you: 7 and 8.** They are held free by the hpREST2 project's queue from
about one hour after the RGDfV explicit ladder starts (roughly 21:40 on 2026-09-13) until you
release them. Take them through the lease if you use that mechanism, or tell the hpREST2 side and
it will keep out of the way. **GPU 0 belongs to TarFlow; 1-6 are running the hpREST2 campaign.
Please do not touch either.**

## What to test, and what each test is for

### 1. REST2 still runs and still exchanges

The baseline. A short ladder — 4 rungs is enough, 2 GPUs oversubscribed — for a few hundred
exchanges. What matters is not that it completes but that:

* `rem.log` has one row per rung per exchange and the state-to-walker map is a permutation at
  every step;
* the acceptance rate is in the range this system gave before 0.5.3;
* `restart.json` records `run_status: completed` and an `exchanges_committed` equal to what was
  asked for.

None of the three fixes should touch this. If it moves, something in the group-file change reached
further than intended.

### 2. Extension across a boundary — the fix that matters

Run a ladder, let it complete, then extend it with `--extend-from`. **Do not pass `-c`.** Before
0.5.3 that aborted every rank at `no coordinates given`; the point of the test is that it now
starts and that what it produces is a continuation rather than a restart.

Check, in the extension's own `restart.json`:

* `resumed_from_step` equals the parent's `steps_completed` exactly — not approximately;
* `steps_completed` equals that plus `extend x exchange_interval_steps`;
* `final_state_to_walker` is a permutation, and differs from the parent's only by the swaps the
  extension actually accepted;
* the parent directory is byte-identical afterwards. An extension that modifies its parent is not
  an extension.

Then the part a unit test cannot reach: **the trajectory must join**. Take the parent's last frame
of each state and the extension's first, and confirm they are continuous — the same configuration
carried across, not a re-equilibration. A restart that silently re-thermalised would pass every
check above.

Also run it **once with `-c` supplied**, to confirm the value is inert: the two runs should give
the same first exchange. If they differ, coordinates are being read on a path that claims not to
read them.

### 3. REST2 with harmonic restraints

The hpREST2 method runs a REST2 ladder with flat-bottom torsion restraints on every rung, and that
combination is what this engine will be asked for next. Two things to establish:

* a ladder with `nmropt=1` and a DISANG file runs, and the restraint energy appears in the output
  rather than being silently dropped;
* **the restraint cancels from the exchange criterion when it is the same on both rungs.** With
  one force constant on every rung, `W` appears identically in `u_i` and `u_j` and drops out of
  `log alpha`. So a restrained ladder and an unrestrained one, from the same seed and the same
  start, should accept the *same* exchanges. If they do not, either the restraint is not identical
  across rungs or it is entering the acceptance calculation where it should not.

That last test is the valuable one. It is cheap, it is exact rather than statistical, and it
catches the error that would otherwise show up months later as a biased free energy.

## What NOT to do

* Do not merge to `main` on my behalf. The hpREST2 side will do that once you report these pass.
* Do not change the three fixes to make a test pass without saying so — if a test fails, the
  interesting outcome is the failure.
* Do not run on GPUs 0-6.

## Reporting back

For each of the three: what you ran, what you measured, and the numbers. A statement that a test
passed is not the same as the number it produced, and for test 2 and the cancellation check in
test 3 the numbers are the result.
