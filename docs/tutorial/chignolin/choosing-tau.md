# Choosing τ_max for chignolin

**Tested against md-tools `0.5.4`.** Background for [REST2: chignolin](REST2.md). Nothing here is
needed to RUN that page; it is the measurement behind the τ span it uses.

## Why 0.3 and not 0.5

The [paracetamol ladder](../paracetamol/REST2.md) spans τ 0 → 0.5 in four states and accepts 0.223. The same
span over chignolin, in six states, does not work:

| ladder | solute atoms | states | τ span | overall acceptance |
|---|---|---|---|---|
| paracetamol | 20 | 4 | 0 → 0.5 | 0.223 |
| chignolin | 138 | 6 | 0 → 0.5 | **0.010** |
| chignolin | 138 | 6 | 0 → 0.3 | **0.133** |

Both chignolin runs took the same 8 minutes and the same 1935 ns/day per replica — the cost is
identical and only the spacing differs. The middle row is a ladder that completes, reports, and
exchanges almost nothing: at 1% the six states are close to six independent simulations, and the
whole point of REST2 is lost while every file still looks right.

The reason is size. The energy difference between neighbouring states grows with the solute, so the
τ spacing that works for a 20-atom molecule is far too coarse for a 138-atom peptide. Halving the
span halves the gap and acceptance rises thirteenfold.

md-tools **reports acceptance and does not enforce it**, and the ladder is linear in τ with no
spacing optimisation ([REST2](../../openmm_methods/REST2/README.md)). So this is a number to read
every time: if the pairs are much below ~0.1, tighten `tau_max` or add states, and remember that
adding states means adding GPUs.

