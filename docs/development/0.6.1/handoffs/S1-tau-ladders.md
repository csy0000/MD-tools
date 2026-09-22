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
