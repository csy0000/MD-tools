# A2 and B2: the same two TYK2 ladders at lower tau_max

Written by session S1 on branch `work/0.6.1-selection`. **This is post-0.6.1 evidence.** Nothing
here changes the release, the tutorial or the notes; the existing ladders A and B stand as they
are, and these two are additions beside them.

Requested by the user (relayed through S0, 2026-09-22), verbatim: *"for ladder A, try to only
scale up to tau=0.3 for the ligand; for ladder B, try to scale up to tau=0.15 for sidechain +
ligand."*

## What changed, and the proof that it is the only thing

| | ladder A | **A2** | ladder B | **B2** |
|---|---|---|---|---|
| hot region | the ligand instance `:291` | same | ligand + 19 pocket sidechains | same |
| `tau_max` | 0.5 | **0.3** | 0.25 | **0.15** |
| rungs, schedule | 8, `kind: linear` | 8, `kind: linear` | 8, `kind: linear` | 8, `kind: linear` |
| spacing `Δτ` | 0.071429 | 0.042857 | 0.035714 | 0.021429 |

**`selection_sha256` of A2 equals ladder A's, and B2's equals ladder B's.** That identity hashes
the Hamiltonian-determining projection -- hot atoms, owned torsion bonds, CMAP decisions,
protected bonds, the ligand instance with its package and resolved exclusions -- so the two pairs
differ in `tau_max` and in nothing else. It is a stronger statement than "the same configuration
file", which is what a reader would otherwise have to take on trust.

Built before any run, each Hamiltonian a file: `build-top --rest2-scaler` from the registered
dataset's own `built.xml` (md5 identical to the registered copy), never writing into a registered
tree, with `CUDA_VISIBLE_DEVICES=""` so the build's platform probe could not reach a card another
session held. Each `scaler.yaml`'s recorded source digest equals `built.xml`'s, each carries 8
states, and no state file is byte-identical to its predecessor's except state 0, which is tau 0
and must be.

**The spacing law is NOT changed with `tau_max`.** Changing two things would make the result a
measurement of everything at once, which is exactly what the request is not asking for.

## The predictions, registered BEFORE the runs

Both ladders keep `Δτ` at 0.6x of their predecessor's. Treating neighbouring-rung energy gaps as
roughly Gaussian with `σ ∝ Δτ`, and inverting each existing ladder's OWN measured acceptance to
get its `σ` (`acceptance = erfc(σ/2)`), rather than fitting anything new:

| quantity | predicted | the model is WRONG if |
|---|---|---|
| A2 acceptance, overall | **0.72** (from A's 0.549: σ 0.847 -> 0.508) | outside 0.65-0.80 |
| B2 acceptance, overall | **0.60** (from B's 0.388: σ 1.221 -> 0.732) | outside 0.52-0.68 |
| A2 round trips | 150-200 | held with LOW confidence: a diffusive quantity over 2500 exchanges |
| B2 round trips | 90-140 | as above; B2 is expected to gain more than A2 in relative terms |
| ns/day per state | **unchanged, 90-97** | tau changes what is sampled, not what a step costs |
| exchange-energy discrepancy | **the same 0.08-0.13 kT band as A and B** | see below |

**The energy check is predicted to BARELY MOVE, and the reason is the falsifiable part**: `|u|` is
a property of the box, the box is identical, and the discrepancy is mixed-precision accumulation
over 53,030 particles rather than anything about how deeply the Hamiltonian is scaled. So A2 is
predicted INSIDE the G2-6b bound and **B2 near it, possibly outside again** -- a smaller tau is
NOT predicted to rescue ladder B's 1.16x miss.

If B2 instead lands comfortably inside while B misses, that contradicts every hypothesis now on
the table, including the sampling-depth one. **It is to be reported as a contradiction and
examined, never quietly reconciled** (S0, 2026-09-22).

On the throughput row: if either ladder differs by more than ~5%, "background load" is not an
acceptable conclusion on its own -- it is the explanation that is always available. What else was
on the cards has to be established, not assumed.

## What will be measured, in the form the other two ladders used

Acceptance per neighbouring pair and overall; walker round trips with the permutation check;
ns/day per state and aggregate; and the exchange-energy check at the final exchange against
**G2-6b's bound evaluated at each ladder's OWN measured `|u|`**, never a borrowed one, with
G2-6-as-written reported beside it for continuity.

## Status

**BUILT, NOT RUN.** Waiting on cards: the gate lane holds 1-6, card 0 is excluded from ladders by
the user's rule and card 7 is hpREST2's. When they free, hpREST2 is asked first, then four
homogeneous cards, 2 ranks each, MPS in a short-path pipe directory of my own that is deleted on
exit.

These runs may deserve their own registered datasets. That needs the user's approval of the names,
directly, as the first two did.

## Registered before measuring: torsion transitions at the hot rung (2026-09-22)

Acceptance and round trips say the LADDER mixes better. They say nothing about whether the top
rung still does the job it exists for: a ligand at tau 0.3 crosses lower barriers than at tau 0.5,
and that is the cost being traded (S0's point, 2026-09-22). The measurement is a count of
**torsion transitions per ns at the hot rung**, over the scaled central bonds, from the
`solute_state7_prod1.nc` trajectories already on disk. No card is needed.

**Method, fixed before the numbers are seen.** Each scaled central bond gives one proper torsion
(heaviest neighbour either side). Basins are defined ONCE from the pooled dihedral histogram of
all four ladders, so every run is discretised the same way, and a transition is a basin change
that persists at least two frames. The trajectories are sampled at 10 ps, so what is counted is
transitions RESOLVABLE at 10 ps; faster recrossing is invisible to this measurement and the
number is a lower bound in all four runs equally.

**The prediction.** Solute-solute torsion terms carry `(1-tau)^2`, so the hot rung's barriers are
at 25% of unscaled for ladder A (tau 0.5) against 49% for A2 (tau 0.3), and 56% for ladder B
(tau 0.25) against 72% for B2 (tau 0.15). For a representative 5 kT barrier that is a rate ratio
of `exp(-5*(0.49-0.25))` and `exp(-5*(0.72-0.56))`:

| | predicted transitions at the hot rung | wrong if |
|---|---|---|
| A / A2 | A crosses **2-5x more often** than A2 | outside that range |
| B / B2 | B crosses **1.5-3x more often** than B2 | outside that range |
| A2 in absolute terms | still crosses: **more than 1 transition per ns** summed over its scaled bonds | fewer, which would mean tau 0.3 buys mixing by giving up the barrier crossing the ladder exists for |

If A2's hot rung samples as many crossings as A's, the user's instruction is a straight win. If it
samples materially fewer, the trade is real and the recommendation has to say where it lies.


## A2 measured, B2 terminated, and what the two together decided (2026-09-22)

### A2: better mixing, and a hot rung that crosses nothing

**The criterion this page exists to establish, and the only part that transfers to another
system: `tau_max` is set by BARRIER CROSSING at the hot rung; acceptance is set by the RUNG
COUNT.** Everything else here is TYK2's numbers.

**A2's acceptance and A2's zero never appear apart.** At tau 0.5 the ligand's three non-rotor
torsions cross **54 times in 5 ns**; at tau 0.3 they cross **ZERO** times. The aggregate counts
(117 against 66) hide it because a single methyl contributes 63 and 66 of them, and a methyl spin
is not a conformational degree of freedom anyone wants sampled. A reader who meets 0.742 and 326
first has already concluded the shorter ladder is better, so the two numbers travel together here,
in the tutorial, and in any summary.

| quantity | predicted (registered at `6f682c6`) | measured | |
|---|---|---|---|
| acceptance, overall | 0.72, wrong outside 0.65-0.80 | **0.742 -- with ZERO non-methyl hot-rung crossings (A: 0.549 with 54)** | INSIDE |
| worst-pair acceptance | not predicted -- see below | 0.720 (spread 1.06x) | — |
| round trips | 150-200, low confidence | **326 -- transporting configurations the hot rung never generated** | **REFUTED, 1.63x above** |
| ns/day per state | 90-97 | 90.7 (aggregate 725.7, 81m09s) | inside, low edge |
| exchange discrepancy | 0.08-0.13 kT | abs 0.0809, cross 0.0804 | inside |

Ladder A, for comparison: 0.549 overall, 0.483 worst pair (spread 1.25x), 105 round trips,
93.5 ns/day, 0.077/0.093 kT.

**Why the round-trip band failed, and the better model.** I predicted mixing roughly in proportion
to acceptance. Mixing is governed by the WORST pair, not the mean (S0): ladder A's pairs spread
0.483-0.602, a factor of 1.25, so a walker had a place to stick at the hot end; A2's spread
0.720-0.764 is 1.06 and nearly uniform. Round trips respond to the bottleneck disappearing, which
is why they tripled while the mean rose 1.35x. **Worst-pair acceptance is reported beside the
overall figure from here on.**

**The energy check at A2's own `|u| = 286,598 kT`**: bounds 0.1383 and 0.1042, measured 0.0809 and
0.0804, inside by 1.71x and 1.30x, with the wrong Hamiltonian missing by 9,977 kT (95,718x the
bound). Against ladder A the absolute barely moved (+5%) while the cross FELL 14% at half the tau.
Depth is constant here -- both ladders ran 2,500,000 steps -- so this is the one comparison in
which tau is the only variable, and it points away from the sampling-depth story. One pair of
points; not a result.

### The measurement that decided it: torsion transitions at the hot rung

Method as registered above. **Rotors are separated**: a central bond whose either end has no heavy
neighbour besides its partner is a methyl, hydroxyl or ammonium spin, not a conformational degree
of freedom, and counting it drowns everything else.

| | ligand non-rotor | sidechain non-rotor | rotor bonds (excluded) |
|---|---|---|---|
| A, tau 0.5 | **10.82 /ns** (C4-N1 28, C1-C5 24, C11-N3 2) | — | 76 |
| A2, tau 0.3 | **0.00 /ns** -- all three bonds, zero in 5 ns | — | 71 |
| B, tau 0.25 | 3.61 /ns | **250.5 /ns** over 33 bonds | 1,284 |
| B2, tau 0.15 (PARTIAL) | 1.45 /ns | **184.7 /ns** over 33 bonds | 1,006 |

**A2's zeros are physical, not a basin-cut artefact**: its two aryl/amide torsions have circular
standard deviations of 15.2 and 19.6 degrees and span under 90 degrees of arc across the whole
5 ns, where ladder A's same torsions have 55.1 and 53.7 degrees and visit the full circle.

**The aggregate hid this completely.** Including rotors, A2 shows 66 transitions against A's 117 --
a 1.77x ratio that looks like a modest cost. Every one of A2's 66 is a methyl spin. My registered
prediction (A crosses 2-5x more often; A2 still crosses more than 1/ns) is **REFUTED IN
SUBSTANCE**: the ratio landed below the band, and the absolute half held only on a methyl, which
makes holding it worthless. Recorded as refuted rather than claimed on the letter.

### The named pattern, once

Three times in one day a SUMMARY STATISTIC stood in for a distribution it could not represent:
the aggregate transition count over a methyl rotor; the mean acceptance over the worst pair that
governs mixing; and a max over a sample whose N and depth were never stated (G2-6b). Each looked
like a measurement and each hid the quantity that mattered. The rule this suggests: **report the
distribution's governing feature beside any summary -- the bottleneck, the excluded trivial mode,
the N.**

### What it means, and the user's ruling

Acceptance and round trips measure whether the ladder TRANSPORTS configurations; transitions
measure whether the hot rung GENERATES anything worth transporting. A2 is a better transport
system carrying nothing.

**The user ruled: "back to tau 0.5 for A"** (2026-09-22, relayed by S0). Ladder A stands as the
recommended ligand ladder, and **A2 is kept as the evidence for that choice**, not discarded: the
sentence that justifies tau 0.5 to the next reader is "tau 0.3 gives 0.742 acceptance, 326 round
trips and zero non-methyl transitions in 5 ns".

The design rule it implies: **tau_max is set by barrier crossing at the hot rung, a physical
criterion; acceptance is then a matter for the RUNG COUNT.** Ladder B's problem was never its
tau_max -- 8 rungs to 0.25 gave a worst pair of 0.358 -- and the fix for that is more rungs at the
same tau, not a colder top rung.

### B2, terminated at 83% by the user's instruction

**The user ruled: "we can terminate the tau=0.15 ladders because 0.15 doesn't give enough
scaling. Also remove the data of the tau_max=0.15 REST2 simulations."** B2 was stopped at **2072
of 2500 exchanges (83%)** and its data removed. These numbers are what the partial run showed and
are **PARTIAL, not a result**:

* acceptance overall 0.616, worst pair 0.578, spread 1.12x (my registered prediction was 0.60,
  band 0.52-0.68 -- inside, on a partial run);
* hot-rung transitions as tabulated above, over 4.13 ns.

**And it answered the question it was run for, which is why the numbers are kept.** Sidechain chi
torsions and ligand amide/aryl torsions do NOT behave alike: from tau 0.25 to 0.15 the sidechains
lost only 26% of their crossings (250.5 -> 184.7 /ns) while the ligand's stiff torsions were
already nearly silent at 0.25 (3.61 /ns) and the ones that cross changed identity at 0.15. So the
recommendation is **per region, not global**: chi barriers are low enough to keep crossing at
small tau, while a ligand's amide and aryl-carbonyl torsions need a genuinely hot rung. The user's
judgement that 0.15 "doesn't give enough scaling" is consistent with the ligand side of that,
which is the side the tutorial's ladder exists for.


## B3: twelve rungs to tau 0.4 -- predictions registered BEFORE the run (2026-09-22)

**The user's instruction**, verbatim: *"Use the GPUs to run 12-rungs tau_max=0.4 REST2"*, replacing
the terminated tau 0.15 ladder.

The region is ladder B's exactly -- `selection_sha256` equals ladder B's, checked -- and the start
is ladder B's own `eq_3.xml` (md5 6a986690...). `Δτ = 0.036364` against B's `0.035714`: **the rung
spacing is within 2% of B's while the top rung is 60% hotter.** That is the point of the design:
if spacing governs acceptance and tau governs barrier crossing, B3 should keep B's acceptance and
gain its transitions.

| quantity | predicted | REFUTED if |
|---|---|---|
| acceptance, overall | **0.41** | outside 0.35-0.48 |
| worst-pair acceptance | **0.38** (B's worst was 0.92 of its mean) | outside 0.32-0.45 |
| round trips | **15-45** -- possibly FEWER than B's 45 despite better acceptance | outside that range |
| hot-rung sidechain transitions | **300-450 /ns** (B: 250.5) | below B's figure |
| hot-rung LIGAND non-rotor transitions | **5-12 /ns** (B: 3.61, A at tau 0.5: 10.82) | at or below B's 3.61 |
| ns/day per state | 88-97, aggregate ~1,100 | the per-rank configuration is unchanged |
| exchange discrepancy | 0.08-0.13 kT, inside the G2-6b bound at B3's own `\|u\|` | — |

**Why acceptance is predicted slightly ABOVE B's rather than equal to it.** The neighbouring-rung
energy gap scales as `d[(1-tau)^2]/dtau = 2(1-tau)Δτ`, so it depends on WHERE in tau the pair
sits, not only on `Δτ`. B3 spans 0-0.4 (mean `1-tau` = 0.8) against B's 0-0.25 (mean 0.875), so at
equal spacing its gaps are about 9% smaller and its acceptance a little higher. An acceptance that
lands exactly on B's 0.388 would mean that second-order effect is absent; one outside 0.35-0.48
would mean spacing does NOT govern acceptance, which is the assumption the whole "fix acceptance
with rung count" rule rests on.

**Why round trips may FALL while acceptance rises.** Round trips are a random walk over the
ladder, and the walk is longer: 12 rungs against 8 needs roughly `(12/8)^2 = 2.25x` as many
accepted steps to cross. At B's acceptance that predicts about `45/2.25 = 20`. **More rungs buy
acceptance and cost diffusion**, and if that is right it is a second design trade to state
explicitly rather than a disappointment.

**The falsifiable heart of the run** is the transition row. The hot rung at tau 0.4 has barriers at
36% of unscaled against B's 56%, so it must cross MORE than ladder B did. A B3 that mixes well and
still crosses nothing would mean the sidechain and ligand barriers need more than tau 0.4, and the
whole "set tau by barrier crossing, fix acceptance with rungs" rule would need rethinking rather
than tuning.

**Configuration**: 12 ranks, 2 per card on cards 1-6, MPS with `CUDA_DEVICE_ORDER=PCI_BUS_ID` on
the daemon, placement verified by nvidia-smi before launch (six client contexts on cards 1-6,
card 0 untouched). The daemon serves 1-8 so that hpREST2 can run its own MPS diagnostic on 7-8
under the same server -- one server per user per node, so a second daemon cannot serve it.


## B3 measured: four of six predictions refuted, and the design rule survives (2026-09-22)

12 rungs, tau_max 0.4, ladder B's region and ladder B's start, 12 ranks on six cards, 91m04s.

| quantity | predicted (`c77f6c2`) | measured | |
|---|---|---|---|
| acceptance, overall | 0.41, refuted outside 0.35-0.48 | **0.373** (B: 0.388) | inside -- but my REASON was wrong |
| worst-pair acceptance | 0.38, refuted outside 0.32-0.45 | **0.341** (B: 0.358) | inside |
| round trips | 15-45 | **2** (B: 45) | **REFUTED, far below** |
| sidechain transitions | 300-450 /ns | **264.3 /ns** (B: 237.7) | **REFUTED, below** |
| LIGAND non-rotor transitions | 5-12 /ns | **23.05 /ns** (B: 3.81; A at tau 0.5: 10.42) | **REFUTED, far above** |
| ns/day per state | 88-97 | **81.5** (aggregate 978.3) | **outside, below** |

**The design rule HOLDS, and this was its test.** Δτ within 2% of ladder B's gave acceptance
0.373 against B's 0.388 and a worst pair of 0.341 against 0.358 -- spacing governs acceptance,
across a 60% change in tau_max. My prediction of a small RISE from the `2(1-tau)Δτ` factor was
wrong in direction: the acceptance fell slightly instead. The rule "acceptance is set by the rung
count" survives; my second-order correction to it did not.

**State-space diffusion, all four ladders, measured from the exchange records** (states² per
exchange; crossing time is `(N-1)²/2D` exchanges, and `K/τ_cross` says how many ladder crossings
the 2500-exchange run had room for):

| | N | D | crossing time | K/τ_cross | round trips |
|---|---|---|---|---|---|
| A, tau 0.5 | 8 | 0.2401 | 102 | 24.5 | 105 |
| A2, tau 0.3 | 8 | **0.3248** | 75 | 33.1 | 326 |
| B, tau 0.25 | 8 | 0.1696 | 144 | 17.3 | 45 |
| B3, tau 0.4, 12 rungs | 12 | 0.1709 | 354 | 7.1 | 2 |

**Round trips: refuted, and the underlying quantity was not.** State-space diffusion per exchange
is `D = 0.1709` states²/exchange for B3 against `0.1696` for ladder B -- **identical**. Per-step
transport did not degrade at all. What changed is the ladder got longer, and crossing time scales
as `N²`: 144 exchanges for B's 7 gaps against 354 for B3's 11, a factor of 2.46 that `(11/7)² =
2.47` predicts exactly. Round trips fell by 22x on a 2.4x change, because a round-trip COUNT is an
extreme-value readout that collapses once the crossing time approaches the run length
(`K/τ_cross`: 17.3 for B, 7.1 for B3). **Report `D` and `K/τ_cross` beside round trips** -- the
fourth instance today of a summary statistic exaggerating what it summarises, and the first one
where the underlying measurement was completely healthy.

**The finding: a hot pocket lets the LIGAND cross.** Ladder B and B3 share a region and differ
only in tau, so this comparison is clean: **3.81 /ns at tau 0.25 against 23.05 /ns at tau 0.4**, a
6x gain in exactly the stiff aryl and amide torsions that A2 showed dying. B3's hot rung also
crosses them **twice as often as ladder A's does at tau 0.5** (10.42 /ns) -- that comparison is
NOT clean, since A differs in region as well as tau, but it is the strongest hint here: heating
the pocket around the ligand appears to help the ligand itself, presumably because a cold pocket
cages it.

**Sidechain chis are saturated and were never the problem**: 237.7 /ns at tau 0.25 against 264.3
at tau 0.4, +11% for a 60% rise in tau_max, where the ligand gained 6x. Chi barriers are low
enough to cross at any tau in this range. **So the per-region answer is: tau_max must be set by
the LIGAND's torsions, and the sidechains come along for free at whatever tau that dictates.**

**Throughput, 81.5 against the 90.7-93.5 of the 8-rung ladders**, is outside my band and I will
not name one cause. Two candidates, neither established: twelve ranks synchronise at every
exchange where eight did, so the barrier is wider; and hpREST2 ran an MPS throughput diagnostic on
cards 7-8 through the same server during B3, and an MPS server is one process multiplexing every
client of a user. A third possibility is that both contributed. Distinguishing them needs a repeat
on an idle server, which is not worth a card today.


### B3's energy check, and what it does to the four-run picture

At B3's own `|u| = 286,783 kT` the G2-6b bounds are 0.1383 and 0.1043. Measured over **144 values**
(12 states, the largest sample of the four): absolute **0.0894**, INSIDE by 1.55x; cross
**0.0996**, INSIDE by 1.05x -- barely. The wrong Hamiltonian misses by 12,215 kT, 117,149x the
bound.

| run | region | tau_max | N | cross discrepancy | against the bound |
|---|---|---|---|---|---|
| A | ligand | 0.5 | 64 | 0.093 | inside, 1.12x |
| A2 | ligand | 0.3 | 64 | 0.080 | inside, 1.30x |
| B | ligand + pocket | 0.25 | 64 | **0.121** | **OUTSIDE, 1.16x** |
| B3 | ligand + pocket | 0.4 | 144 | 0.0996 | inside, 1.05x |

**This weakens every explanation offered for ladder B's miss, including mine.** B3 shares B's
region, box, start and sampling depth, and has 2.25x the sample -- a max over a LARGER sample
should be larger, not smaller -- yet it comes in 18% below B's. Nor does tau explain it: within
the ligand region the higher tau gave the higher discrepancy (A 0.093 > A2 0.080), and within the
pocket region the higher tau gave the LOWER one (B3 0.0996 < B 0.121). Opposite directions in the
two pairs.

So across four runs the cross discrepancy wanders between 0.080 and 0.121 with no clean dependence
on tau, on N, or on depth, and ladder B's 0.121 looks like the high end of that spread rather than
a property of ladder B. **That is the strongest evidence yet that the ESTIMATOR is the defect**:
`max |dU|` over whatever sample a run happens to produce is too noisy to sit under a 1.16x
verdict. G2-6c's job is now concrete -- a quantile or RMS at a stated N and depth would give a
statistic whose spread is known, and none of these four runs would be judged by whether its
noisiest pair landed above a line.

Nothing here is repaired retroactively: ladder B's row stays as measured and as FAILED, and B3's
row is INSIDE by 1.05x, which is not a margin anyone should lean on.

## Open: a cheap experiment that would apportion B3's throughput shortfall

B3 ran at 81.5 ns/day per state against the 8-rung ladders' 90.7-93.5, and the write-up names two
candidates without choosing: a wider exchange barrier (twelve ranks synchronise where eight did)
and MPS server contention (hpREST2 ran a throughput diagnostic through the same server during
B3). Its own measurement quantifies the cross-effect in the other direction -- my twelve ranks
cost its light client 13% at four processes and 27% at eight -- so contention is real, but a light
client's penalty does not transfer to my ranks.

**The experiment that separates them** (hpREST2's suggestion, 2026-09-22): run the SAME ladder at
6 ranks and at 12 ranks through an otherwise-idle server. A shared-server penalty and an exchange
barrier scale differently with rank count, so the difference between those two isolates them. It
needs cards and an idle machine and nothing else, and nothing in this page depends on the answer.

**Also open**: ladder A's region at tau 0.5 with 12 rungs -- whether the acceptance A2 bought can
be had without giving up the ligand's barriers. B3 makes it likely (same spacing, same
acceptance, hot rung unchanged) but likely is not measured, and the hot rung would be ladder A's
exactly, so `selection_sha256` and the tau 0.5 states already exist.
