# Analysis

**Only md-tools' own analysis tools are described here.** RMSD, MBAR, WHAM and kinetic analysis
are standard methods with a literature; they are used in the tutorials and explained where they
are used. This page covers the three things this package does that you will not find elsewhere,
all under `md_tools.analysis`:

| | | needs |
|---|---|---|
| [`t_hdbscan`](#t_hdbscan) | a density partition of a torsional ensemble | scikit-learn |
| [`t_mi`](#t_mi) | mutual information between torsions, with its own bias measured | numpy |
| [`t_symmetry`](#t_symmetry) | which of those clusters are the same conformer | rdkit |

They are independent: none imports another's estimator, and `import md_tools.analysis` loads none
of them. See [Installing](../../install.md#5-optional-the-analysis-environment) for the extra.

---

## `t_hdbscan`

A torsional ensemble is points on a TORUS, not in a Euclidean space, and that governs everything.
Two frames at +179° and −179° are neighbours; any method that treats the angle as a real number
puts them maximally far apart and cuts a single basin in half at the periodic seam. So every
distance here is computed in the `(cos θ, sin θ)` embedding, where Euclidean distance is a
monotone function of the chordal distance on the torus.

On that embedding the partition is HDBSCAN: a density-based method that finds basins of arbitrary
shape and labels low-density frames as noise rather than forcing them into a cluster.

**Three things it does not claim.**

*A cluster is a density basin, NOT a metastable state.* Nothing in this module looks at time. Two
density peaks separated by a sampled region are two clusters whatever the barrier between them,
and a long-lived state that is geometrically diffuse may be one cluster or none. Metastability
needs dynamical evidence this estimator does not have.

*Seed agreement is PRECISION, not accuracy.* `draw_agreement_` says the cluster count was stable
under resampling. It does not say a small state was seen.

*With non-uniform weights and `resampling=False`, only the POPULATIONS are reweighted.* HDBSCAN
cannot weight its density estimate — the core distance is a k-th nearest neighbour distance, a
RANK statistic, so there is no `sample_weight` to pass — and the partition is therefore still
fitted on the sampled ensemble. `resampling=N` draws N frames with probability proportional to
the weights (Madow systematic, WITHOUT replacement, because duplicate frames give zero distances
and a core distance of zero) and fits the target distribution itself.

### `allow_single_cluster` defaults to True, unlike sklearn

With it False, HDBSCAN's excess-of-mass selection refuses the root of the condensed tree, so a
**unimodal ensemble cannot come back as one state** — it is split into two or three with a
plausible mass split and nothing saying one state was never a candidate. Measured here: a 4000-frame single Gaussian
splits into 2.

The cost of True points the other way: a genuinely multi-state ensemble can collapse into one
cluster when the between-basin density does not beat the root's stability. True is still the
better default because the two errors are not equally visible. One state reported where there are
three is a single population that looks wrong and prompts a check; three reported where there is
one is several plausible populations that look like a result. **Over-splitting is the harder
error to notice.**

`summary()` carries `single_cluster_excluded_by_construction` either way, so which case you are in
is reported rather than inferred — and when the answer IS one cluster, a caveat tells you to check
it against the torsion marginals first.

---

## `t_mi`

Mutual information between pairs of torsions: how much knowing one tells you about the other,
in nats, from a 2D histogram on the torus.

**The plug-in histogram estimator is biased UPWARD**, and that is the whole reason this is not
three lines of numpy. An independent pair does not return zero; it returns a small positive
number that reads as a weak coupling. Measured on a paracetamol trajectory, the pair with the
LARGEST raw MI was the one that is entirely bias:

| pair | raw | debiased | excess over null |
|---|---|---|---|
| omega–aryl | 0.0334 | **+0.0206** | **+9.6 σ** |
| aryl–phenol | **0.0462** | +0.0013 | +0.4 σ |
| omega–phenol | 0.0117 | −0.0027 | −1.2 σ |

So the bias is MEASURED, per pair, against a null built by shuffling ONE axis — which destroys
the dependence while keeping both marginals and the binning exactly as they were. With `n_null=0`
the debiased value is `None`, never silently equal to the raw one.

**No verdict field.** A boolean "significant" would collapse +9.6 σ and +0.4 σ into the same
answer, and +0.4 σ is exactly where a reader needs the number rather than a token. Three
normalisations are returned because they answer different questions: `I/sqrt(HiHj)` (symmetric,
the usual NMI), `I/min(Hi,Hj)`, and the asymmetric `I/Hi`, which is the one a redundancy rule must
use — dropping torsion *i* is only safe if *i* is determined by what remains.

### Across the whole set

`analyse` asks which pairs are coupled among all of them, which is a different problem: *d*
torsions means *d(d−1)/2* simultaneous comparisons, so a handful of "significant" pairs is the
expected yield of pure noise. It sweeps the binning and the origin, bootstraps over BLOCKS rather
than frames (a correlated trajectory has far fewer independent samples than frames), and applies
a Benjamini–Hochberg false-discovery-rate correction.

Its verdicts are three-valued and **none of them says "independent"**: a pair where no dependence
was detected at this resolution and this sample size has not been shown independent.
`dependence_graph` gives the connected components over supported pairs — a CANDIDATE analysis
group, not proof of irreducible many-body coupling, and a missing edge rules nothing out, because
pairwise MI is blind to dependence that appears only in three or more variables jointly.

---

## `t_symmetry`

Some clusters are the same physical conformer seen through a permutation of chemically identical
atoms. Paracetamol's phenyl ring has a two-fold axis: flipping it maps one basin onto another with
no physical change at all. Merging those is right; merging two genuinely different conformers
because their populations happen to match is wrong, and everything here is arranged to prevent
the second.

**What counts as evidence:** a graph AUTOMORPHISM of the molecule, found by RDKit, respecting
element, bond order, formal charge, isotope and specified stereochemistry — and then agreement of
the two clusters' JOINT distributions, in both directions.

**What does not:** equal populations, similar one-dimensional histograms, equal canonical ranks, a
force-field multiplicity, a bin count, or an unconstrained spatial reflection.

### Why a permutation is applied whole

In a para-disubstituted benzene the two-fold flip moves BOTH attachment torsions together. Folding
each torsion into its own period independently would also identify the configuration where only
one substituent was rotated — a different relative orientation, and a different conformer. So an
operation acts on the whole descriptor vector as one object, never torsion by torsion.

### When the transformed torsion is not in the table

An operation maps `1-3-4-5` to `1-3-4-10`: same central bond, different outer atom. If that
quadruplet is among the selected torsions its stored column is reused (a dihedral is exactly
reversal-invariant, so `D-C-B-A` counts). Otherwise it is **recomputed from coordinates**, and if
there are none the operation is reported UNRESOLVED with the missing quadruplets named.

It is never approximated by an idealised 180° shift. Ring distortion and pyramidalization move the
outer atoms independently of the central bond, so the shift is not a constant — and substituting
one would hide exactly the deviations the comparison is meant to be sensitive to.

### Deciding equivalence

The statistic is the **energy distance** on the periodic embedding: a genuine distance between
distributions, zero if and only if they are equal, using the JOINT configuration of every torsion
at once, and needing no binning — a binned test on *d* torsions needs exponentially many frames to
fill the grid.

It is symmetric, so it can stay small when one cluster is a SUBSET of another — a narrow basin
nested inside a broad one. **Coverage is therefore asked in each direction separately** and both
must pass, which is what distinguishes "the same distribution" from "this one fits inside that
one". A pair that passes energy but only one direction of coverage is reported as a partial
relationship rather than merged.

**Tolerances are calibrated from the data, not invented.** Each cluster is split into its first
and second half BY FRAME INDEX — temporally separated, so the two samples carry the
autocorrelation a random split would hide — and the energy distance between halves is measured.
That is what two samples of the same distribution produce here. A split-half distance is a FLOOR
on the sampling error: it says nothing about a slow motion neither half sampled, and that caveat
travels with the tolerance.

**Transitivity is not assumed.** Under a loose threshold A~B and B~C can both pass while A and C
are plainly different. A connected component is only a candidate group: every member is then
re-tested against the group's reference directly, and one that fails is split back out.

### Populations

Masses SUM across a merge, once. Nothing is multiplied by the symmetry order — the order says how
many labels the ensemble was split into, not how much probability it holds. Noise keeps its own
label and its own mass, and merged masses plus noise equal the original total exactly.

### What it does not establish

That two clusters are related by a graph automorphism says their conformations are **structurally**
equivalent. It does NOT establish that their thermodynamic populations should be equal. An
atom-specific restraint, or a REST2 scaling region covering one side and not the other, breaks the
equivalence while leaving the graph intact — declare those with `broken_by=[...]` and they are
recorded on every accepted pair.

---

## Where these are used

* [Paracetamol: clustering, pruning and summary](../../tutorial/paracetamol/analysis/index.md)
* [Paracetamol: which torsions move together](../../tutorial/paracetamol/analysis/index.md)
