# Analysis

**Only md-tools' own analysis tools are described here.** RMSD, MBAR, WHAM and kinetic analysis
are standard methods with a literature; they are used in the tutorials and explained where they
are used. This page covers what this package does that you will not find elsewhere, all under
`md_tools.analysis`:

| | | needs |
|---|---|---|
| [`t_hdbscan`](torsional-clustering/hdbscan.ipynb) | a density partition of a torsional ensemble, symmetry-first by default | scikit-learn, rdkit |
| [`cut_clustering`](torsional-clustering/cell-partitioning.ipynb) | a partition by barrier cuts on one-dimensional free-energy profiles; every frame labelled | rdkit (on the symmetry-first route) |
| [`t_mi`](#t_mi) | mutual information between torsions, with its own bias measured | numpy |
| [`t_symmetry`](#t_symmetry) | the older post-hoc route: which clusters of an existing partition are the same conformer | rdkit |

They are independent: none imports another's estimator, and `import md_tools.analysis` loads none
of them; `t_hdbscan` reaches the symmetry enumeration only on its default route, lazily. See [Installing](../../install.md#5-optional-the-analysis-environment) for the extra.

---

## Torsional clustering

Two ways to partition a torsional ensemble, on one preprocessing. Each page is a notebook run on the
tutorial data, and every parameter is shown changing a result.

| page | what it covers |
|---|---|
| [Torsion preprocessing](torsional-clustering/torsion-preprocessing.ipynb) | the input contract; the symmetry enumeration and its canonicalisation; the closed descriptor and its weights; the symmetric distance HDBSCAN uses; the symmetry-invariant coordinates the cell partitioning cuts; one labelling for drawing, a gauge, a crystal as reference; symmetry-corrected RMSD |
| [HDBSCAN](torsional-clustering/hdbscan.ipynb) | `t_hdbscan`: `mass_floor`, `min_samples`, `allow_single_cluster`, the vote, `resampling`, the torsion set; `refine_by_rmsd`, `hierarchical_clustering`; migrating an older call |
| [Cell partitioning](torsional-clustering/cell-partitioning.ipynb) | `cut_clustering`: `kT_cut`, `mass_floor`, `min_p2p`, `kT_step`, `n_bins`, `z_min`, `gauge`; the splitting tree |

```python
from md_tools.analysis import t_hdbscan, cut_clustering, TorsionDefinitions

definitions = TorsionDefinitions(quads=((0, 1, 3, 4), (1, 3, 4, 5), (6, 7, 8, 17)),
                                 names=("omega", "aryl", "phenol"),
                                 indexing="zero-based", units="radian")
fit = t_hdbscan(torsions,                     # (n_frames, 3) radians, in that column order
                molecule=mol,                 # RDKit Mol with explicit H (the SDF)
                torsion_definitions=definitions,
                coordinates=traj,             # mdtraj.Trajectory: coordinates, unit cell, topology
                atom_mapping=mapping)         # trajectory index of each reference atom

fit.fit().labels_                  # HDBSCAN: 0, 1, ... by population; -2 density noise; -1 ambiguous
cut_clustering(fit)["labels"]      # cell partitioning: 0, 1, ... by mass; every frame labelled
```

**Which one.** They answer different questions and agree where both are clear (two clusters on
paracetamol; the large clusters of alanine dipeptide and c(RGDfV)).

* HDBSCAN finds **density basins** in the joint torsion space, so it separates two basins that overlap
  in every one-dimensional projection. It fits an exact distance matrix, at most 10 000 frames
  (`resampling=N` above that), and leaves frames it cannot place unassigned, as noise or ambiguous.
* The cell partitioning finds **regions bounded by barriers in one-dimensional profiles**, reads every
  frame (300 000 on c(RGDfV)), gives every frame a label, and states its threshold as a barrier in kT. It
  does not separate basins that overlap in every projection.

**A cluster from either is not a metastable state.** Nothing in either looks at time.

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

* [Torsion preprocessing](torsional-clustering/torsion-preprocessing.ipynb), [HDBSCAN](torsional-clustering/hdbscan.ipynb), [Cell partitioning](torsional-clustering/cell-partitioning.ipynb)
* [Alanine dipeptide: conformations by cell partitioning](../../tutorial/ALA/analysis/clustering.ipynb)
* [Paracetamol: torsional clustering](../../tutorial/paracetamol/analysis/clustering.ipynb)
* [c(RGDfV): conformations of a cyclic peptide](../../tutorial/rgdfv/analysis/clustering.ipynb)
* [Paracetamol: the old and new workflows compared](../../tutorial/paracetamol/analysis/clustering-comparison.ipynb)
* [Paracetamol: which torsions move together](../../tutorial/paracetamol/analysis/index.md)
* [Archived 0.6.4: cluster, then merge by symmetry](../../tutorial/archived/0.6.4/paracetamol/clustering.ipynb)
