# Analysis

**Only md-tools' own analysis tools are described here.** RMSD, MBAR, WHAM and kinetic analysis
are standard methods with a literature; they are used in the tutorials and explained where they
are used. This page covers the three things this package does that you will not find elsewhere,
all under `md_tools.analysis`:

| | | needs |
|---|---|---|
| [`t_hdbscan`](#t_hdbscan) | a density partition of a torsional ensemble, symmetry-first by default | scikit-learn, rdkit |
| [`t_mi`](#t_mi) | mutual information between torsions, with its own bias measured | numpy |
| [`t_symmetry`](#t_symmetry) | the older post-hoc route: which clusters of an existing partition are the same conformer | rdkit |

They are independent: none imports another's estimator, and `import md_tools.analysis` loads none
of them; `t_hdbscan` reaches the symmetry enumeration only on its default route, lazily. See [Installing](../../install.md#5-optional-the-analysis-environment) for the extra.

---

## `t_hdbscan`

### The default workflow: symmetry first

```text
enumerate molecular symmetry -> symmetry-aware torsional distance -> HDBSCAN -> classify frames
                                                                     (same distance, 18 of 20)
```

```python
from md_tools.analysis import t_hdbscan, TorsionDefinitions, verify_atom_mapping

definitions = TorsionDefinitions(quads=((0, 1, 3, 4), (1, 3, 4, 5), (6, 7, 8, 17)),
                                 names=("omega", "aryl", "phenol"),
                                 indexing="zero-based", units="radian")
fit = t_hdbscan(torsions,                     # (n_frames, 3) radians, in that column order
                molecule=mol,                 # RDKit Mol with explicit H (the SDF)
                torsion_definitions=definitions,
                coordinates=traj,             # mdtraj.Trajectory: coordinates, unit cell, topology
                atom_mapping=mapping).fit()   # trajectory index of each reference atom
fit.labels_            # 0, 1, ... by population; -2 density noise; -1 ambiguous
fit.summary()          # populations, unassigned masses, vote, symmetry, metric, caveats
```

A torsion is measured through four NAMED atoms, and a molecular symmetry renames atoms without
moving any. Paracetamol's ring flip renames C5 as C10, so the `aryl` torsion measured to C5
becomes the torsion measured to C10: the same configuration, a different number. The distance
between two frames therefore has to be independent of which equivalent naming each is written in:

```text
Phi(theta) = (..., sqrt(w_j) cos theta_j, sqrt(w_j) sin theta_j, ...)
d_sym(x, y) = min over g, h in G of || Phi(g x) - Phi(h y) ||
```

Frames that differ only by a relabelling are then at distance **zero before HDBSCAN builds its
density**, so symmetry-related basins share a cluster during the fit, before the minimum cluster
size is applied. The older workflow fitted the labellings as separate clusters and merged them
afterwards; it is still available (below) and the two are compared on paracetamol in the
[comparison notebook](../../tutorial/paracetamol/analysis/clustering-comparison.ipynb).

**Relabelling is not motion.** The enumerations are alternative descriptions of one observation.
They are never appended to the data as extra frames, which would multiply frame counts and
statistical weights by the group order.

### The input contract

Angles alone cannot establish a molecular symmetry, so the default route needs:

| input | what it is | when |
|---|---|---|
| `torsions` | `(n_frames, n_selected)` stored angles, in the order of the definitions | always |
| `molecule` | an RDKit molecule with explicit hydrogens | always |
| `torsion_definitions` | a `TorsionDefinitions` indexing that molecule | always |
| `coordinates` | an `mdtraj.Trajectory`, or an `(n, n_atoms, 3)` array with `box_vectors` (or `periodic=False`) and a `topology` | when a symmetry maps a selected torsion onto one that is not in the table |
| `atom_mapping` | the trajectory index of each reference atom; never defaulted | with `coordinates` |

Missing information raises with the call that fixes it. Nothing guesses a symmetry, and nothing
falls back to ordinary clustering on its own. The mapping is checked twice: by graph (elements and
every bond among the mapped atoms) and by geometry (each stored column must agree with the same
quadruplet recomputed from the coordinates, to 1e-3 rad by default). Molecules split across the
periodic box are made whole along their bonds, for orthorhombic and reduced triclinic cells, before
any dihedral is computed.

### Enumerating the symmetry

`enumerate_symmetry` matches the molecule against ITSELF with RDKit (`uniquify=False`,
`useChirality=True`), which returns every graph automorphism of the whole molecule -- rings,
methyl groups, anything. It works on a copy with atom-map numbers cleared, re-verifies every match
as an element- and bond-order-preserving bijection, checks that every specified stereocentre and
stereo double bond keeps its CIP label, puts the identity first, and asks for one match more than
its cap (200 000): a search that hits the cap raises `IncompleteSymmetryEnumeration` instead of
being used. Permutations that act identically on the descriptor collapse to one operation:
paracetamol has 12 automorphisms (3! methyl-hydrogen orders x the ring flip) and 2 distinct
operations, because no selected torsion contains a methyl hydrogen.

### When the one-sided minimum is exact

If every operation acts on the descriptor as an ISOMETRY of the embedding,

```text
d_sym(x, y) = min over g in G of || Phi(x) - Phi(g y) ||
```

which costs |G| evaluations instead of |G|^2. That holds exactly when (1) every operation maps the
set of descriptor torsions onto itself, so it acts as a permutation of columns, and (2) the metric
weights are constant on every orbit of that permutation group. **An arbitrary selection satisfies
neither.** Paracetamol's `aryl` (1-3-4-5) maps to 1-3-4-10, which nobody selected, so the flip does
not even act on the three selected columns. `SymmetricTorsionDescriptor` therefore:

* **closes** the selection: every image of a selected quadruplet becomes a column, named
  `<torsion>@<atoms>` (`aryl@1-3-4-10`, `phenol@9-7-8-17`), recomputed from coordinates -- never
  approximated by a 180 degree shift, which on paracetamol would be wrong by up to 25 degrees;
* **normalises** the weights: each orbit carries the summed weight of the torsions selected in it,
  split equally among its columns (paracetamol: `omega` 1, the four ring-dependent columns 0.5
  each), so the ring orientation counts once and not twice; selected torsions in one orbit must
  have equal weights, or the constructor refuses;
* **verifies** closure under composition and the isometry when it is built, and raises if either
  fails. `exhaustive_distance` computes the two-sided minimum, and the test suite checks the two
  agree on small fixtures, including a molecule with a three-fold axis, where an operation and its
  inverse differ.

The minimising operation is returned for inspection (`distance_matrix(..., return_operation=True)`,
`distance`, `pair_table`); ties go to the earlier operation. What is NOT done: choosing a
"canonical" labelling per frame and taking ordinary Euclidean distances between representatives,
which breaks pairs that sit on either side of the canonicalisation boundary.

### One distance, everywhere

`d_sym` is used for the HDBSCAN fit, the nearest-neighbour vote, the density threshold and the
query density check, and the medoid tie-break of `representatives()`. It is a minimum over
relabellings and not Euclidean in any fixed embedding, so it is handed to HDBSCAN as an **exact
dense matrix** (`metric="precomputed"`) and never forced into a Euclidean KD-tree. One copy is
`8 n^2` bytes -- 122 MiB at the 4000 frames of the paracetamol tutorial -- and sklearn holds about
three. **Above `max_precomputed_frames` (10000) the fit refuses** and names the subset route,
`resampling=N`: N frames are drawn in proportion to the weights, fitted, and every frame is then
classified with the same distance. The metric is never swapped to make a fit cheaper.

### Assigning frames: 18 of 20

Every frame is classified by its `k = 20` nearest CLUSTERED training frames, itself excluded. It
joins the winning cluster when

```text
vote_fraction = n_winner / k_effective >= 0.90        (equality accepted)
```

so 18 of 20 assigns and 17 of 20 does not. `k_effective` is 20, or the number of eligible training
frames when there are fewer, and it is reported per frame (`fit.k_effective_`). Neighbours at
equal distance are ordered by index, so the decision does not depend on the numpy build.

Two kinds of unassigned frame, with different codes and different meanings:

* **-2, density noise** -- the frame's `min_samples`-th neighbour is farther than that of any
  clustered training frame, under the same distance. Takes precedence.
* **-1, ambiguous** -- fewer than 90 % of the neighbours agree. This describes the sample around
  the frame; it is **not** evidence of a free-energy barrier. (`AMBIGUOUS_LABEL`; the former name
  `BARRIER_LABEL` is kept as an alias of the same value.)

`noise_mass`, `ambiguous_mass` and their sum `unassigned_mass` are reported separately. Populations
are weighted sums over every frame; the density the fit sees is that of the fitted frames,
unweighted unless `resampling=N`.

### Migrating an existing call

| before | now |
|---|---|
| `t_hdbscan(tors)` | refused: pass `molecule`, `torsion_definitions` (and `coordinates`, `atom_mapping`), or `symmetry=False` |
| `t_hdbscan(tors)` for the old numbers | `t_hdbscan(tors, symmetry=False, vote_rule="legacy-margin")` |
| `min_vote=0.9` (a MARGIN, k = 15) | refused unless `vote_rule="legacy-margin"`; the new threshold is `min_vote_fraction` (k = 20). A margin is never reinterpreted as a fraction: 18 of 20 against two votes is a fraction of 0.90 and a margin of 0.80 |
| `summary()["min_vote"]` | `summary()["vote"]` (`rule`, `k`, `k_effective_min/max`, `min_vote_fraction`); `min_vote` is still written on the legacy-margin rule |
| `noise_mass_` (both codes) | `noise_mass_` is DENSITY noise only; add `ambiguous_mass_`, or read `unassigned_mass_` |
| name `"barrier"` for -1 | `"ambiguous"` |
| `representative_by_vote(fit)` | unchanged call; it now uses the fit's own distance and vote (`fit.representatives()`) |
| `multiplicities=[...]` with symmetry | refused: per-torsion folding is what the symmetry-first distance replaces |
| cluster, then `TorsionSymmetry` merge | the default route needs no merge; the merge remains available for the legacy route |

**Structural, not thermodynamic.** A graph automorphism says two configurations are the same
conformer under another naming. It does not make the populations of the two namings equal: an
atom-specific restraint, an atom-indexed parameter or a REST2 region covering one side breaks the
energetic symmetry and leaves the graph unchanged. Declare such a thing with `broken_by=[...]`; it
is recorded in the summary.

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

**The post-hoc route, kept for the legacy workflow and for comparison.** The default `t_hdbscan`
enumerates the symmetry before clustering and needs no merge. `TorsionSymmetry` takes an EXISTING
partition and decides which of its clusters to merge; it is what the
[archived 0.6.4 tutorial](../../tutorial/archived/0.6.4/paracetamol/clustering.ipynb) used.

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

* [Paracetamol: symmetry-first clustering](../../tutorial/paracetamol/analysis/clustering.ipynb)
* [Paracetamol: the old and new workflows compared](../../tutorial/paracetamol/analysis/clustering-comparison.ipynb)
* [Paracetamol: which torsions move together](../../tutorial/paracetamol/analysis/index.md)
* [Archived 0.6.4: cluster, then merge by symmetry](../../tutorial/archived/0.6.4/paracetamol/clustering.ipynb)
