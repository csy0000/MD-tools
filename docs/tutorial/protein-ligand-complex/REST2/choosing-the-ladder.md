# Choosing τ_max and the rung count: what four ladders showed

**Tested against md-tools `0.6.1`.** Background for
[REST2 on a protein–ligand complex](README.md). Nothing here is needed to RUN that tutorial: this
is the evidence for the settings it uses, the datasets they produced, and the two numbers on them
that mislead.

!!! note "Four ladders were run on 2026-09-21 and 2026-09-22; every number here is copied from their files"
    RTX 3080s, 2 ranks per card under MPS, CUDA mixed precision, md-tools 0.6.1 on branch
    `work/0.6.1-selection`. The system is the prepared TYK2 fixture with `ejm_31`. Route A and
    the comparison ladder used four cards; route B, at twelve rungs, used six.

    **One check on this page did not pass**: the recorded exchange energies were recomputed
    against the saved states and came out at 0.077–0.121 kT, against a 0.05 kT tolerance that had
    been calibrated on a system thirty times smaller. The tolerance does not transfer between
    system sizes and is being replaced by a derived one; until that is measured, this check is
    **unvalidated on systems of this size**. It is recorded rather than quietly widened.

!!! warning "Why route B is 12 rungs at τ 0.4, and not the 8 rungs at τ 0.25 that were tried first"
    Four ladders on this complex settle it. **Set τ_max by whether the hot rung crosses the
    barriers you care about; set acceptance by the number of rungs.** For this ligand:

    | ladder | τ_max | rungs | acceptance | ligand torsion crossings at the hot rung |
    |---|---|---|---|---|
    | A | 0.5 | 8 | 0.549 | 10.4 /ns |
    | A2 | 0.3 | 8 | 0.742 | **0.0 /ns** |
    | B | 0.25 | 8 | 0.388 | 3.8 /ns |
    | B3 | 0.4 | 12 | 0.373 | **23.1 /ns** |

    At τ 0.3 the ligand's aryl and amide torsions **never cross in 5 ns** — while acceptance and
    round trips both look their best of any ladder here. A ladder can transport configurations
    beautifully and have a hot rung that generates nothing new, and the acceptance figure cannot
    see it. (Counting every scaled bond would have hidden this too: a methyl on this ligand spins
    freely at any τ and supplies most of the raw count. Methyls and hydroxyls are excluded above.)

    Raising the pocket ladder to τ 0.4 over 12 rungs gives **6x the ligand crossings of B at the
    same rung spacing and the same acceptance** (0.373 against 0.388). The pocket sidechains were
    never the constraint: their χ torsions cross 238 /ns at τ 0.25 and 264 /ns at τ 0.4, a gain of
    11% where the ligand gained 6x. **τ_max is set by the ligand; the sidechains come along.**

    The cost is transport. Per-exchange diffusion is identical (0.171 against 0.170 states²
    /exchange), but a 12-rung ladder takes `N²` longer to traverse — 354 exchanges against 144 —
    so B3 completed 2 round trips where B completed 45. More rungs buy acceptance and pay for it
    in traversal, so use the fewest rungs that give workable acceptance at the τ_max your barriers
    demand.


## 5. What these runs measure

| quantity | route A (ligand, 8 rungs, τ 0.5) | route B (pocket, 12 rungs, τ 0.4) | comparison (pocket, 8 rungs, τ 0.25) |
|---|---|---|---|
| hot atoms (of 4,701 solute atoms) | 32 | 193 | 193 |
| scaled torsion central bonds | 4 | 52 | 52 |
| scaled CMAP terms | 0 | 0 | 0 |
| **ligand torsion crossings, hot rung** | **10.4 /ns** | **23.1 /ns** | **3.8 /ns** |
| sidechain χ crossings, hot rung | — | 264 /ns | 238 /ns |
| neighbouring-pair acceptance | 0.483–0.602, overall 0.549 | 0.344–0.415, overall 0.373 | 0.358–0.408, overall 0.388 |
| worst pair | 0.483 | 0.341 | 0.358 |
| walker round trips (5 ns) | 105 | 2 | 45 |
| diffusion, states²/exchange | 0.240 | 0.171 | 0.170 |
| ns/day per state | 93.5 | 81.5 | 96.3 |
| ns/day aggregate | 748 (4 cards) | 978 (6 cards) | 771 (4 cards) |
| wall clock | 83 min | 91 min | 81 min |

Crossings exclude methyl and hydroxyl rotors, which spin freely at any τ and would otherwise
supply most of the count.

**The finding: τ_max buys barrier crossing and rungs buy acceptance, and the two are separable.**
Route B heats six times the region of route A and still crosses the ligand's torsions twice as
often, at a τ_max 0.2 lower, because the rung count — not a colder top rung — is what pays for the
acceptance. Against the 8-rung comparison at τ 0.25 it is a 6x gain in ligand crossings for a
10 ns/day cost and an acceptance that is the same to within 4%.

!!! warning "Two numbers on this table that do not mean what they look like"
    **Round trips: 2 for route B against 45 for the comparison.** Per-exchange diffusion is
    identical (0.171 against 0.170) — transport did not degrade at all. A 12-rung ladder simply
    takes `N²` longer to traverse (354 exchanges against 144), and a round-trip *count* collapses
    once the crossing time approaches the run length. Judge mixing by the diffusion constant and
    by the worst pair, not by a count that a longer ladder cannot help but lose.

    **ns/day: 81.5 for route B.** Twelve ranks synchronise at every exchange where eight did, and
    another job was sharing the same MPS server during that run. The
    two causes are not separated here, so do not read 81.5 as the price of twelve rungs.

### Every run here is a registered dataset

Every figure in the table above is read from these, not retyped from a log:

```text
$MD_DATA/2026/tutorials/tyk2-ejm31-rest2-ligand-8x5ns                110 files, 545.8 MB   route A
$MD_DATA/2026/tutorials/tyk2-ejm31-rest2-ligand-pocket-12x0.4-5ns    108 files, 693.9 MB   route B
$MD_DATA/2026/tutorials/tyk2-ejm31-rest2-ligand-pocket-8x5ns         117 files, 546.0 MB   the COMPARISON
```

**The third is not a route you should follow.** `...-pocket-8x5ns` is the 8-rung τ 0.25 pocket
ladder that route B replaced, kept because it is the measurement that justifies route B's twelve
rungs: without it, "12 rungs at 0.4" is advice rather than a result. A registered dataset is
write-once, so it stays where it is and is labelled rather than removed.

Only the 12-rung dataset carries the source structure and the ligand package, so it is the only
one `export-reference` can read; see the warning in section 6.

Each holds the whole tree a reader needs to check the claim -- `build/` with the scaled states and
`scaler.yaml`, the equilibration chain, and the ladder's own `REST2.nc`, `restart.json` and
per-state trajectories -- and each verifies against its own inventory:

```bash
md-openmm data-register --verify-only -idata <the dataset> \
    -project_name tutorials -data_name tyk2-ejm31-rest2-ligand-8x5ns -year 2026
```

!!! warning "Their energy check is a recorded FAIL, and the datasets say so"
    Gate G2-6 recomputes a ladder's recorded exchange energies against its saved states. Every run
    here FAILS it as originally written — 0.077/0.093, 0.106/0.121 and 0.089/0.100 kT against a
    0.05 kT tolerance calibrated on a system thirty times smaller. Against the size-aware
    replacement `dU <= 3*K*sqrt(|u|)`, validated at a third system size, route A and route B are
    inside on both channels (route B's cross margin is only 1.05x) and **the 8-rung comparison
    ladder is outside by 1.16x**.

    Across four ladders the cross discrepancy wanders between 0.080 and 0.121 kT with no clean
    dependence on τ, on sample size or on sampling depth — a larger sample even gave a *smaller*
    maximum — so the estimator, a max over whatever sample a run produces, is the thing being
    fixed rather than any one ladder. A deliberately wrong Hamiltonian is caught by four to five
    orders of magnitude more, so this is a question about a tolerance, not about the scaling. Each
    dataset's notes carry its own numbers.

### How fast is eight rungs on four cards, really

**The naive expectation:** one rank alone does 330 ns/day on this system, two ranks share a card,
four cards — so something near 4 × 330 / 2 per rung.

**The measurement:** **93.5 ns/day per rung**, 748 ns/day aggregate. Each of the two ranks on a
card gets 57% of what one rank alone gets, so a shared card delivers about **1.13×** its
single-rank throughput, not 2×. Part of that is the sharing and part is the exchange barrier:
every rank waits for the slowest rung at every exchange, so a ladder runs at the pace of its
slowest card — which is also why the cards must be the same model.

Plan from the measured number, not from the single-rank one. Multiplying a single-rank figure by
the card count overestimates a ladder by about a factor of two.

**How the cost divides.** Three 100 ps probes, same system, same equilibrated start. **This table
is for comparing configurations with each other — do not size a campaign from its absolute
numbers; see the warning directly below it.**

| configuration | ranks per card | ns/day per rung | ns/day aggregate |
|---|---|---|---|
| one rank, no ladder at all | — | 330 | — |
| 4 rungs on 4 cards | 1 | 209.9 | 839.5 |
| 8 rungs on 4 cards | 2 | 117.6 | 940.8 |
| 8 rungs on 2 cards | 4 | 69.1 | 553.1 |

!!! warning "These probe numbers are 26% too high for planning"
    The 8-on-4 probe reads 117.6 ns/day per rung. The 5 ns production ladder of the same shape
    sustained **93.5**. A 100 ps probe does not pay the accumulated cost of a long run, so a
    campaign sized from this table comes out about a quarter short. Take the relative scaling from
    here and the absolute number from a production run.

The ladder machinery — exchange barriers, per-state trajectories, checkpoints — costs 36% before
any sharing (330 → 209.9). A second rank on a card gives back 1.12× that card's aggregate
throughput, not 2×.

**A fourth rank per card is a loss, not a smaller gain.** Eight rungs on two cards delivers 553.1
ns/day aggregate, *below* the 839.5 of four rungs on four cards and far below the 940.8 of eight
on four. Two ranks per card is the most this system rewards; three is untested and four is worse
than not sharing at all.

The comparison the campaign asks for is between A and B: what heating the pocket as well as the
ligand costs in acceptance, and whether it buys sampling the ligand-only ladder does not reach.

