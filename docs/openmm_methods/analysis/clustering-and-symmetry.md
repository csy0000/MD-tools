# Symmetry-aware clustering and the torsion representation

This page explains four things the torsional analysis does, in the order a frame meets them: how a
symmetric torsion is **duplicated and weighted**, how the **distance is minimised over
permutations**, how **HDBSCAN is used with a 1 % floor**, and how a frame is **written in one
labelling** (the Dirichlet-domain rule). The tool reference is [Analysis](README.md); the results are
on the [paracetamol](../../tutorial/paracetamol/analysis/symmetry-alignment.ipynb) and
[c(RGDfV)](../../tutorial/rgdfv/analysis/clustering.ipynb) pages.

!!! note "What is released"
    The distance, the clustering and the weights are in 0.6.5. The canonicalization of section 1b,
    `THDBSCAN.density_peak`, `THDBSCAN.align_all`, `THDBSCAN.assign_reference`,
    `SymmetricTorsionDescriptor.align_to_reference` and `nearest_labelling`, `reference_assignments` and
    the symmetry-corrected RMSD are new in 0.6.5.1, and `allow_single_cluster` now defaults to False.

## The problem

A torsion is measured through four **named** atoms, and a molecular symmetry renames atoms without
moving any. Paracetamol's phenyl ring flip renames C5 as C10, so the same configuration is
`aryl = -16` degrees when measured through C5 and `aryl = 164` degrees when measured through C10.
Two consequences follow for anything built on the stored columns:

* a **distance** between frames that ignores this calls one configuration two, and a clustering
  built on it returns one cluster per naming;
* a **histogram** of one stored column is the sum of one conformer written twice, with peaks half a
  turn apart that are not two states.

Sections 1-3 fix the first, section 4 the second. In all four, **relabelling is not motion**: no atom
moves, and no frame is added or dropped.

## 1. Duplicating torsion bonds, with weights

The molecule is matched against itself with RDKit (`uniquify=False`, `useChirality=True`). Every
match is a graph automorphism: an element-, bond-order- and stereo-preserving permutation of the atoms.
c(RGDfV) has 147 456 of them and paracetamol has 12 (c(RGDfV) has 36 864 if the drawn structure is
compared instead of the canonical form, see [section 1b](#1b-canonicalization-atoms-the-drawing-keeps-apart)).
Most do nothing to the torsions that were selected, so permutations that act identically on the
descriptor collapse to one **operation**:

| molecule | automorphisms | distinct operations on the selected torsions | what they are |
|---|---|---|---|
| paracetamol | 12 | 2 | identity, the ring flip (the 3! orderings of the methyl hydrogens move no selected torsion) |
| c(RGDfV), 17 torsions | 147 456 | 4 | identity, the Phe ring flip, the Val methyl swap, both |
| c(RGDfV), 19 torsions (adding Asp `chi2`, Arg `chi5`) | 147 456 | 16 | those four exchanges, the carboxylate oxygens and the guanidinium nitrogens, in every combination |

An operation maps a selected quadruplet to another quadruplet with the same central bond and a
different outer atom: `aryl` (1-3-4-5) becomes 1-3-4-10. If nobody selected 1-3-4-10, it is added.
The descriptor is **closed**: every image of a selected torsion becomes a column of its own, named
`<torsion>@<atoms>`, and its value is **recomputed from coordinates**. It is never approximated by a
180 degree shift: ring distortion moves the outer atoms independently of the central bond, and on
paracetamol the shift is wrong by up to 25 degrees.

Closing makes the operations act as **permutations of columns**, which is what makes the distance in
section 2 exact. It also duplicates a torsion, and a duplicated torsion would count twice. So each
**orbit** of columns (the columns an operation can exchange) carries the summed weight of the
torsions selected in it, **split equally among its columns**:

| | column | weight |
|---|---|---|
| paracetamol | `omega` | 1 |
| | `aryl`, `aryl@1-3-4-10`, `phenol`, `phenol@9-7-8-17` | 0.5 each |
| c(RGDfV) | the 15 backbone phi, psi, omega | 1 each |
| | `Val:chi1`, `Val:chi1@4-3-1-2`, `Phe:chi2`, `Phe:chi2@7-8-9-14` | 0.5 each |

The ring orientation therefore counts once, as it should. Two torsions selected in one orbit must
have equal weights, or the constructor refuses: nothing says which one the orbit should follow. The
group closure and the isometry property of section 2 are **checked when the descriptor is built**,
and it raises if either fails.

Each column contributes `sqrt(w) (cos θ, sin θ)` to the embedding `Φ(θ)`, so a torsion is compared as
a point on a circle: `-π` and `π` are the same place.

## 1b. Canonicalization: atoms the drawing keeps apart

A graph with localised bond orders and charges draws a carboxylate as `C=O` and `C-O-`. RDKit's
self-match compares those, finds the two oxygens different, and returns no permutation that
exchanges them: the torsion measured to one oxygen is never folded onto the torsion measured to the
other, half a turn away. The same holds for the two terminal nitrogens of a guanidinium.

`enumerate_symmetry` therefore **canonicalises** the molecule before it matches (`canonicalize=True`,
the default; `resonance=` is the former name and still works, with a deprecation warning).
The canonical form is the resonance hybrid: the charge of each atom and the order of each bond are averaged over the resonance structures that
RDKit's `ResonanceMolSupplier` generates, and two atoms with the same hybrid values, the same
element and the same neighbours are one atom. A molecule with no delocalised group gets back exactly
its own integers, so nothing else changes.

Three rules keep this from over-reaching:

* **Aromatic atoms and bonds keep their own values.** The supplier returns an aromatic ring
  Kekulised, and averaging the alternating 1 and 2 would break the ring's own symmetry (phenol would
  lose its ring flip). A ring is already one delocalised system.
* **The structure count is a limit that refuses.** A result of 1000 structures cannot be told from a
  truncated one, and is not used (`max_structures`).
* **Every returned permutation is re-verified** against the hybrid values, as before against the
  drawn ones.

RDKit does not generate the alternative drawings of a hypervalent group, so one rule is added: the
**terminal oxygens (or sulfurs) of a sulfur, phosphorus or chlorine centre** are one atom, whichever
way the group is drawn.

Measured automorphism counts, drawn structure against hybrid (hydrogens explicit):

| group | example | drawn | hybrid |
|---|---|---|---|
| carboxylate | acetate | 6 | 12 |
| carboxylate on a ring | benzoate | 2 | 4 |
| guanidinium | N-methylguanidinium | 24 | 48 |
| amidinium | acetamidinium | 24 | 48 |
| nitro | nitromethane | 6 | 12 |
| nitrate | nitrate | 2 | 6 |
| carbonate | carbonate | 2 | 6 |
| sulfonate (rule) | methanesulfonate | 12 | 36 |
| phosphate monoester (rule) | methyl phosphate | 12 | 36 |
| unchanged | acetic acid, acetamide, urea, phenol, paracetamol, phenolate, an asymmetric azide | = | = |

### An example: propionate

One torsion to a carboxylate oxygen, `C0-C1-C2-O3`, in `CCC(=O)[O-]` (atoms C0 C1 C2 O3 O4).

```python
import types
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from md_tools.analysis import TorsionDefinitions, SymmetricTorsionDescriptor, enumerate_symmetry

mol = Chem.AddHs(Chem.MolFromSmiles("CCC(=O)[O-]"))
AllChem.EmbedMolecule(mol, randomSeed=3)
torsions = TorsionDefinitions(quads=((0, 1, 2, 3),), names=("chi",),
                              indexing="zero-based", units="radian")

enumerate_symmetry(mol, canonicalize=False).order     # the drawn structure
enumerate_symmetry(mol).order                         # canonicalised (the default)

for flag in (False, True):
    d = SymmetricTorsionDescriptor(mol, torsions, canonicalize=flag)
    print(flag, d.names, list(d.column_weights), len(d.operations))

# one frame, and the same frame with the two oxygens named the other way round
conf = mol.GetConformer(); rdMolTransforms.SetDihedralRad(conf, 0, 1, 2, 3, 0.4)
xyz = conf.GetPositions()[None]
swap = list(range(mol.GetNumAtoms())); swap[3], swap[4] = 4, 3
atoms = [types.SimpleNamespace(index=a.GetIdx(), element=types.SimpleNamespace(symbol=a.GetSymbol()))
         for a in mol.GetAtoms()]
top = types.SimpleNamespace(atoms=atoms, n_atoms=len(atoms), bonds=[
    (atoms[b.GetBeginAtomIdx()], atoms[b.GetEndAtomIdx()]) for b in mol.GetBonds()])
kw = dict(atom_mapping=list(range(mol.GetNumAtoms())), periodic=False, topology=top)

d = SymmetricTorsionDescriptor(mol, torsions)
a = d.closed_angles(None, coordinates=xyz, **kw)[0]
b = d.closed_angles(None, coordinates=xyz[:, swap, :], **kw)[0]
d.distance(a, b)                                      # (distance, minimising operation)
d.nearest_labelling(np.vstack([a, b]), reference=a)   # a crystal that offers both assignments
```

```text
automorphisms, drawn structure : 12
automorphisms, resonance hybrid: 24
canonicalize=False: columns ('chi',), weights [1.0], operations 1
canonicalize=True : columns ('chi', 'chi@0-1-2-4'), weights [0.5, 0.5], operations 2
frame, O named one way : [  22.9 -156.2] ('chi', 'chi@0-1-2-4')
same frame, O swapped  : [-156.2   22.9]
distance between them  : 0.0 (minimising operation 1)
crystal candidates -> chosen assignment 0 angles [  22.9 -156.2]
```

The same frame is `chi` = 22.9 degrees named one way and -156.2 degrees named the other, a difference
of 179 degrees, which is the two-peak split of a carboxylate torsion in a histogram of the stored
column. Without `canonicalize` the descriptor has one column and one operation and nothing relates
the two; with it they are at distance zero.

**Other places the same thing happens**, not measured here but by the same mechanism: thiocarboxylates,
sulfates and phosphonates (the last two through the rule), and the charged side chains of Glu and the
C terminus. Enolates and 1,3-dicarbonyl anions are exchanged through resonance only when the
drawing is not already symmetric. A carboxylic acid is **not** one: `C=O` and `C-OH` are different
atoms, and so are the two oxygens of an ester.

**What this does not cover.**

* **Tautomers are not resonance.** Moving a hydrogen (imidazole N-H, keto-enol) changes the graph,
  and a protomer or tautomer is a different molecule, not a relabelling.
* **The hybrid weights the structures equally.** RDKit lists the structures, not their
  contributions. For a group whose two sides contribute unequally, an equal average could make two
  atoms look alike that are not; none of the groups above is such a case.
* **The legacy post-hoc route** (`TorsionSymmetry`, `t_symmetry`) still compares the drawn
  structure. The default `t_hdbscan` route, the descriptor and `enumerate_symmetry` use the hybrid.
* **A larger group.** Resonance multiplies the automorphism count (c(RGDfV): 36 864 to 147 456,
  and about 14 s to enumerate), and the 200 000 cap applies as before.

On c(RGDfV) the 17 torsions of the tutorial are not moved by the carboxylate and guanidinium
exchanges, so their four operations and their clusters are unchanged. Adding Asp `chi2` and Arg
`chi5`, which those exchanges do move, folds each of their two peaks half a turn apart into one, and
the descriptor has 16 operations; see the
[c(RGDfV) page](../../tutorial/rgdfv/analysis/clustering.ipynb#11-canonicalization-the-carboxylate-and-the-guanidinium).

## 2. The distance, minimised through the permutations

```text
d_sym(x, y) = min over g in G of || Φ(x) - Φ(g y) ||
```

`g y` is `y` with its columns permuted by operation `g`. The one-sided minimum is exact when every
operation is an isometry of the embedding, which holds exactly when each operation permutes the
columns and the weights are constant on every orbit; section 1 guarantees both and checks them. The
two-sided minimum over `g, h` is kept as `exhaustive_distance`, and the test suite checks the two
agree, including on a molecule with a three-fold axis where an operation and its inverse differ.

Cost: `|G|` matrix products, because every embedded point has the same squared norm `W` (the total
weight), so `||a - b||² = 2W - 2 a·b`. For c(RGDfV) that is 4, not 36 864.

**The operation acts on the whole row.** In para-disubstituted benzene the flip moves `aryl` and
`phenol` together. Folding each torsion into its own period independently would also identify the
configuration where only one of them turned, which is a different conformer. Ties go to the earlier
operation, with the identity first.

**What is not done.** Choosing a canonical labelling per frame and taking ordinary Euclidean distances
between the representatives breaks pairs that sit on either side of the canonicalisation boundary.
Relabelling is never appended to the data as extra frames, which would multiply the frame counts and
the statistical weights by the group order.

Because `d_sym` is a minimum over relabellings and not Euclidean in any fixed embedding, it is
passed to HDBSCAN as an exact dense matrix (`metric="precomputed"`): `8 n²` bytes per copy. Above
10 000 frames the fit refuses and names the subset route (`resampling=N`).

## 3. HDBSCAN, with a floor of 1 %

HDBSCAN is fitted on the `d_sym` matrix, so frames that differ only by a relabelling are at distance
zero **before** the density is built, and symmetry-related basins share a cluster during the fit.
Nothing is merged afterwards.

Two numbers set the density estimate, and both are **fractions of the fitted frames**:

| | rule | paracetamol, n = 4000 | c(RGDfV), n = 4002 |
|---|---|---|---|
| `min_cluster_size` (the floor) | `ceil(0.01 n)` | 40 | 41 |
| `min_samples` (the core-distance k) | `round(0.01 n)` | 40 | 40 |

* **The floor is a mass, not a count.** A cluster holding under 1 % of the ensemble cannot be a
  cluster; its frames become noise. A fraction asks the same question at every dataset size; the
  earlier absolute default of 500 frames loosened the floor to 0.17 % at 300 000 frames.
* **`min_samples` is pinned separately.** sklearn lets `min_samples` follow `min_cluster_size`, and
  `min_samples` sets the core distances, so raising the floor alone changes the density estimate
  instead of filtering clusters out of a fixed one. Measured on 15 000 hot c(RGDfV) frames, a 5 % floor
  with `min_samples` following turned all frames into noise; pinned, the same floor returned the two
  clusters above 5 %.
* **Tying `min_samples` to `n` has a cost**, measured on alanine dipeptide: it smooths away a marginal
  state (a 1.44 % basin was absorbed). `min_samples=25` restores the pinned behaviour.

Then every frame, fitted or not, is classified with the **same distance** by an absolute vote of its
20 nearest clustered neighbours (18 of 20 assigns). A frame whose `min_samples`-th neighbour is
farther than that of any clustered frame is density noise (label `-2`); one whose neighbours disagree
is ambiguous (`-1`, which is not evidence of a barrier). Clusters are numbered by population, `0` is
the largest, and `allow_single_cluster` defaults to False, as in sklearn: with it a unimodal ensemble cannot come
back as one state. See [Analysis](README.md#t_hdbscan) for both.

**A cluster is a density basin, not a metastable state.** Nothing here looks at time.

**The choice of torsions moves the answer.** On c(RGDfV) the 17 columns (15 backbone plus the two
symmetric side-chain columns) give two clusters, 0.834 and 0.145, whichever way `allow_single_cluster`
is set. All 24 heavy-atom torsions give one cluster when it is True and two when it is False: the Arg
chi angles and the Asp carboxylate rotation are close to uniform, and they dilute the density that
separates the backbone conformers. The torsion set is an input, and the
[c(RGDfV) page](../../tutorial/rgdfv/analysis/clustering.ipynb#10-the-choice-of-torsions-matters) shows
the table.

## 4. The Dirichlet-domain rule: one labelling for the histogram

The clustering needs no choice of labelling. A histogram does. Each frame is written in the labelling
**nearest one reference frame**:

```text
g*(x) = argmin over g in G of || Φ(ref) - Φ(g x) ||          frame x is shown as g*(x) x
```

This picks one point from each orbit of frames, the one in the **Dirichlet domain** of the
reference: the region of the torus closer to the reference than to any of its images. The reference
is the **density peak of the largest cluster**, an actual frame (the one with the smallest core
distance), not a mean, since a circular mean of a multimodal torsion can describe no configuration.

```python
fit = t_hdbscan(torsions, molecule=mol, torsion_definitions=definitions,
                coordinates=traj, atom_mapping=mapping).fit()
ref = fit.density_peak(0)                      # densest frame of the largest cluster
al = fit.align_all(per_cluster=True)           # every frame in the labelling nearest `ref`
al["selected"]                                 # (n_frames, n_selected) angles, new labelling
al["angles"]                                   # every closed column, images included
al["operations"], al["margin"]                 # the operation used, and how safe the choice was
```

**The same rule as a function of the descriptor**: `descriptor.align_to_reference(closed, ref_row)`
returns the aligned angles, the operation per frame, the distance, the distance with no
relabelling, and the margin.

**The margin** is the second-best operation's distance minus the best. A small margin means the frame
sits on the wall between two labellings, and the choice for that frame is arbitrary.

### One reference or two levels

With one reference, every frame is compared with the one frame. A cluster **far from it** can sit
astride the wall: both of its labellings are about equally far from the reference, so the choice
flips from frame to frame and the cluster is cut in two. On paracetamol this is exactly the case.
Cluster B differs from A by half a turn in the aryl, or, written through the flip, by half a turn in
the phenol; B is equidistant from A's peak under either labelling, and 58 % of its frames have a
margin below 0.1. Its `aryl` histogram shows two peaks.

`align_all(per_cluster=True)` uses **two levels**. Each cluster's frames are aligned to the cluster's
own density peak, and only that one peak is written in the labelling nearest the global reference.
Each cluster is then compact, and the one arbitrary choice left is a single frame per cluster, which
fixes where the cluster is drawn and nothing about the cluster itself. On paracetamol no frame has a
margin below 0.1 any more.

On c(RGDfV) the two agree to the precision of the figures (margin below 0.1 for 2-4 % of frames in
either), because the clusters differ in the backbone and are not equidistant. The two-level
alignment is therefore the safe default, and it costs one extra alignment per cluster.

### Fold one angle, leave the partner free?

A natural shortcut for a **symmetric pair**, such as the two ring torsions: fold one angle into a
window of width `2π/N` around the reference value, and leave the other unfolded. This is a valid
fundamental domain **only when each frame has exactly one image in the window**. That holds when the
operation acts on the orbit as a rotor, N columns spaced `2π/N` apart, and nothing sits near the window
edge. The test is cheap: for every frame count the **distinct** images of the first column inside the
window.

| orbit (c(RGDfV) frames) | no image in window | exactly one | two | median spacing of the pair |
|---|---|---|---|---|
| `Phe:chi2` / its ring image | 0.05 % | 99.93 % | 0.02 % | 175 degrees |
| `Val:chi1` / its methyl image | 37.4 % | 62.6 % | 0 | 122 degrees |

The phenyl ring passes: its two columns are half a turn apart, so one of them always lands in a
window of width π. The valine isopropyl fails: its two methyl torsions are 120 degrees apart, and a
window of width π catches neither for 37 % of the frames. **The fold-one shortcut is not a general
rule**; the Dirichlet rule has no such condition and covers it.

The two are not the same cut either. They are different fundamental domains of one group action:
the Dirichlet domain is defined by the distance of the **whole row** to the reference, so the
operation chosen for a frame can depend on its other torsions, while fold-one looks at one column. On
paracetamol fold-one places both clusters at `aryl` near 0 and separates them by the phenol; the
two-level rule places cluster B at `aryl` near 180 degrees, where its own peak is.

The second angle of a ring pair is not a free torsion. It is `aryl + 180 degrees + δ`, where `δ` is
the small ring distortion. The flip swaps the two columns, so it maps `δ` to `-δ`: the sign of `δ`
is a property of the labelling. On the paracetamol trajectory `δ` is symmetric about zero (mean 0.1
degrees, standard deviation 7.1), so relabelling does not change its histogram.

### A gauge on one column, for a symmetry that moves several columns

Folding is valid column by column only when an operation moves **one** column. A ring flip does not:
on paracetamol it replaces `aryl` by `aryl + 180 degrees` **and** `phenol` by `phenol + 180 degrees`
in the same step, and every one of its operations moves both or neither. Folding each into a half-circle
on its own would identify `(0, 0)`, `(0, 180)`, `(180, 0)` and `(180, 180)`, where the flip identifies only
`(0, 0)` with `(180, 180)` and `(0, 180)` with `(180, 0)`: two real conformers merged.

What is valid is to **fix the gauge of one column and let the others follow it**. Choose a half-open
window `[lo, hi)` for a gauge column `c`. Each frame is written with the operation that puts `c` in the
window, and every other column goes with it, because the operation acts on the whole row. The
columns it moved together keep their full circle: `phenol` is not folded, it is carried.

**Centre the window on the peak.** A window fixed at `[-90, 90)` degrees puts the cut wherever zero
happens to fall, and a cluster whose `aryl` peak is near it is split between the two ends. Give a
**width** instead of the limits, and the window is centred on the reference's own value of the column
(the density peak): the bulk of the cluster sits in the middle, the cut falls half a width away in the
gap, and the histogram is one continuous bump. A half-turn is the width for a two-fold ring flip.

```python
al = fit.align_all(per_cluster=True, gauge=("aryl", np.pi))     # a half-turn window centred on the peak
al["selected"]                  # aryl in the window, phenol written with it
al["gauge_window"]              # (column, lo, hi) that was used, in radians
al["gauge_satisfied"]           # False where no operation lands aryl in the window
al["gauge_edge_distance"]       # radians from aryl to the nearest edge of the window
```

The same argument is on `descriptor.align_to_reference(closed, ref_row, gauge=...)`, which takes
either `(column, width)` or the explicit `(column, lo, hi)`. The column is a name or an index, the window
must be wider than 0 and at most one turn, and anything else is refused by name. `align_all` resolves
the window **once**, from the global reference, so every cluster is drawn on the same axis.

**What is chosen.** Among the operations that satisfy the window, the nearest to the reference is taken,
so whatever the window leaves free (a carboxylate swap acting on other columns, say) is still settled by
the Dirichlet rule above. With `per_cluster=True` each cluster is aligned to its own peak under the same
window. A full-turn window, or no gauge, gives the plain alignment.

**It is for drawing, and the cut is not hidden.**

* The clustering never sees it. Its distance is the minimum over every operation and has no cut, which is
  why a gauge is not applied *before* the distance: two frames at `aryl = 89` and `91` degrees are one
  conformer, but the window writes them half a turn apart, and a distance on the written rows would split
  a cluster there. A gauge placed with no mass near its edges does not do this, which is what centring on the peak aims
  for and the edge count checks.
* A frame near an edge may fall on either side with noise. `gauge_edge_distance` says how near, and a
  count of the frames below a small threshold belongs beside any figure.
* The two images of a ring torsion are half a turn apart only up to the noise in each, so a frame **on
  the cut** can have both just outside the window. It takes the operation that lands the column nearest the
  window, and `gauge_satisfied` is `False` for it, with `gauge_edge_distance` giving how far outside it is.
  It is never silently moved into the window.
* It needs an operation that can reach the window. For an orbit whose images are not evenly spread (the
  valine isopropyl) a half-circle may be missed by every image; those frames are flagged the same way.

The [paracetamol notebook](../../tutorial/paracetamol/analysis/symmetry-alignment.ipynb#10-a-gauge-on-one-torsion-aryl-in-a-half-circle-phenol-following)
shows it: `aryl` in a half-turn window centred on its peak, `phenol` carried with it, and the frames near the cut counted. It has
not been run on a ligand with a different joint symmetry; the folic-acid ring (two torsions moved together
by a ring flip) is the same structure and has not been tried on a trajectory.

### A crystal structure, or any other reference

A crystal ligand drawn over the histograms follows the same rule. Matching it onto the simulated
molecule gives one **atom assignment per graph match** (the two oxygens of a carboxylate, the two
sides of a ring, a swapped methyl), and each assignment writes the same structure as a different row
of closed angles. Drawing a line per assignment puts several lines on a panel for one structure.

`descriptor.nearest_labelling(candidates, reference)` chooses **one**: the candidate row nearest the
reference under the same distance, taken over the columns measurable in every candidate (a torsion
through a hydrogen is `NaN` in a structure without hydrogens and takes no part). The reference is
the row the histograms are drawn in: the density peak of cluster A for a panel pooling the clusters,
and a cluster's own density peak, already relabelled, for that cluster's row. Frames drawn **as
stored**, in any labelling, keep every assignment, because each is a legitimate place for the crystal.

The [paracetamol notebook](../../tutorial/paracetamol/analysis/symmetry-alignment.ipynb#8-paracetamol-in-a-crystal-one-structure-two-assignments) does this with two PDB entries. `reference_assignments(structure, molecule, descriptor.quads)` returns the candidate rows, one per graph match; for paracetamol there are two for each crystal.

`fit.assign_reference(candidates)` then says which cluster the structure resembles: per cluster the
distance to its density peak, to its nearest frame, and the fraction of the cluster's frames that are
nearer the peak than the crystal is. **It returns no best cluster when the measurable columns cannot
separate two clusters** (`best_cluster` is `None`, with `consistent_with` and `indistinguishable`): two
clusters whose peaks coincide on those columns, in some labelling, differ only in columns a crystal
does not have. On paracetamol that is the case: A and B differ in the hydroxyl hydrogen (`phenol`), so
neither PDB structure can be assigned to either, and the notebook says so.

The crystal's line can still sit far from a cluster's histogram; that is a statement about the
conformation, and the labelling no longer contributes to it.

### What this does and does not establish

* **Where the cut falls.** A rotor and its relatives give a clean picture. A peak that crosses the
  wall appears at both ends of the axis, as a peak at `±π` does on any circle. The cut belongs to the
  display; it is not in the data.
* **The reference changes where a cluster is drawn**, never the clusters, the frames in them or their
  populations.
* **Fixed points.** The operations are permutations of columns, never reflections: matching is
  stereo-aware, so a mirror image is not an automorphism. A permutation that swaps two columns fixes
  the frames where the two are equal, and frames near one are nearly equidistant to two labellings;
  the margin reports them.
* **Structural, not thermodynamic.** A graph automorphism says two configurations are one conformer
  under another naming. It does not make their populations equal: an atom-specific restraint or a
  REST2 region covering one side breaks the energetic symmetry and leaves the graph unchanged
  (`broken_by=[...]`).
* **Canonicalization puts resonance in the graph**, as in section 1b, and the carboxylate and
  guanidinium torsions fold. Atoms that are equivalent only by tautomerism are not.
* **Weights.** Populations are weighted sums over frames and the density the fit sees is that of the
  fitted frames, as in [Analysis](README.md#t_hdbscan).

The idea, minimising over the automorphisms, is the same one symmetry-corrected RMSD uses for atom
coordinates; here it acts on a descriptor of torsions instead.

## Symmetry-corrected RMSD

An RMSD by atom index calls a structure with its carboxylate oxygens named the other way round
different from itself. The correction is the same minimum over the symmetry operations as the
torsional distance, with a superposition in place of the circle embedding:

```text
RMSD_sym(x, y) = min over g in G of  min over rotations R, translations t of  RMSD(x, R g y + t)
```

For one fixed operation `g` the optimal rotation is Kabsch's, so the minimum over operations of the
Kabsch RMSD is the **global** minimum over rotation and permutation together. Nothing alternates and
nothing is greedy.

```python
from md_tools.analysis import (enumerate_symmetry, heavy_atoms, operations_on_atoms,
                               symmetric_rmsd, intercluster_rmsd)

atoms = heavy_atoms(mol)
operations = operations_on_atoms(enumerate_symmetry(mol), atoms)     # c(RGDfV): 16, not 147 456
rmsd = symmetric_rmsd(X[:, atoms], ref[None, atoms], operations)     # (n_frames, 1), same units as X
table = intercluster_rmsd(X[:, atoms], fit.labels_, operations)      # mean and std between and within clusters
```

* **Heavy atoms.** The operations are the whole-molecule permutations as they act on the chosen atoms,
  so the cost is the number of distinct ones: 16 for the 41 heavy atoms of c(RGDfV), 2 for the 11 of
  paracetamol. The atom set must be a union of symmetry orbits, or `operations_on_atoms` refuses.
  Hydrogens are left out by choice: all 79 atoms would need all 147 456 operations.
* **Cost.** About 2.6 us per pair and operation in numpy (covariances for all pairs by matrix
  multiplication, then a batched 3x3 SVD, which dominates). 4000 frames, all pairs and 16 operations
  is roughly 5 minutes on one core: an estimate from that rate, not a run of that size.
* **Between and within clusters.** `intercluster_rmsd` returns the mean and standard deviation for
  every pair of clusters; the diagonal is the spread over distinct pairs inside a cluster. Large
  clusters are subsampled (`max_per_cluster`, seeded) and the frames used are recorded.
* **Compared with other tools** (measured on c(RGDfV), 41 heavy atoms): spyrmsd, which takes the
  minimum over all graph isomorphisms, agrees with this to 0.002 A; cpptraj `symmrmsd`, which fits
  once, assigns each symmetric group by a Hungarian assignment and fits again, stays within 0.004 A on
  heavy atoms but gave an RMSD **below** the valid minimum on 29 of 100 frames (up to 0.07 A) once polar
  hydrogens were added, so it is not used as a reference here. spyrmsd matches atoms by atomic number
  and adjacency only, so it ignores bond orders and chirality; this uses the canonicalised, chirality-
  checked group.
* **Not done.** The exchangeable atoms that nothing else couples (the two carboxylate oxygens, the two
  valine methyls) could be assigned by a small Hungarian problem and only the coupled operations (the
  phenyl flip) enumerated. In a test on 100 frames that matched the exhaustive minimum to 0.0001 A,
  but alternating assignment and fit is exact only for a fixed alignment, so it is not what this does.

## Splitting a cluster that is too wide in space

A torsional cluster is a density basin in torsion space, and nothing in the fit looks at how far apart
its members are in Cartesian space. HDBSCAN's condensed tree can keep two sub-populations together
(a parent more stable than its children), and a small torsion difference can move many atoms. The
symmetry-corrected RMSD within the cluster says when that has happened, and `refine_by_rmsd` acts on
it:

```python
fit = t_hdbscan(torsions, molecule=mol, torsion_definitions=defs,
                coordinates=traj, atom_mapping=mapping).fit()
refined = fit.refine_by_rmsd(traj)                   # cutoff=2.0 A, an mdtraj.Trajectory is read as nm
refined["labels"], refined["cluster_mass"], refined["tree"]
```

For each cluster the mean pairwise RMSD of its frames is compared with `cutoff` (default **2 A**). A
wide cluster is fitted again on its own frames, with the fit's own settings and
`allow_single_cluster=False`, and the split is kept only if the refit returns at least two clusters.
Each child is then examined the same way. The loop ends at a node that is inside the cutoff, smaller
than `min_frames` (default `2 k`, 40), at `max_depth` (4), or where the refit finds no second cluster;
every accepted split shrinks the frames per node, so it is finite. `tree` records one entry per node
examined (frames, RMSD mean and std, and which of those four ended it).

* **Atoms measured.** Every heavy atom of the molecule (`moved_atoms`): a torsion turns everything
  attached beyond its central bond, so the four atoms that define it understate what it displaces.
  `atoms='quadruplets'` measures only the defining atoms, closed under the molecular symmetry
  (`torsion_atoms`), and a list replaces them with a union of symmetry orbits. Hydrogens are not
  measured, so two clusters that differ only in a hydroxyl proton (paracetamol A and B) are neither
  merged nor split by this.
* **The cutoff is the guard.** The refit forbids a single cluster, so a cluster wider than the cutoff
  is divided whenever the refit finds a density structure in it, including a single broad basin: a
  paracetamol basin of mean RMSD 0.064 A was cut into three at a cutoff of 0.05 A and left whole at
  0.2 A. Choose the cutoff as the size of a difference you would call a different structure.
* **A mean can hide a minority.** A cluster that is 90 % one structure and 10 % another 4 A away has a
  mean pairwise RMSD of about 0.7 A. `statistic=0.95` (a quantile) tests the tail instead.
* **Frames the refit drops.** A frame the refit calls noise or ambiguous takes that code and leaves the
  cluster; a split moves mass out of a cluster and never into one it was not part of.
* **What it does not change.** The fit, its `predict` and its vote are untouched: new frames are still
  classified by the first-level partition. The returned labels are for the frames the fit was made from.
* **Tested on** a synthetic paracetamol ensemble (cis and trans amide, merged into one cluster by hand
  because the fit separates them) at cutoffs of 0.03-0.3 A, the scale of that molecule. Not run on
  c(RGDfV) or any registered trajectory; whether 2 A is the right cutoff for a cyclic peptide is not
  established here.

## Hierarchical clustering: two rounds, a thermodynamic cut, bits, mass order

One HDBSCAN fit can keep two basins together (a parent more stable than its children), and lowering
`min_samples` to separate them mostly produces noise clusters. `hierarchical_clustering` takes a
fitted symmetry-first `THDBSCAN` and applies, on every node (the whole ensemble first):

```python
from md_tools.analysis import t_hdbscan, hierarchical_clustering, plot_hist, show_compound

fit = t_hdbscan(torsions, molecule=mol, torsion_definitions=defs,
                coordinates=traj, atom_mapping=mapping, mass_floor=0.01).fit()
result = hierarchical_clustering(fit, traj, gauge=("t3 aryl-CO", np.pi),
                                 torsion_null_test=40, kT_cut=1.6)
result["labels"], result["n_clusters"], result["rounds"], result["cuts"], result["bits"]
plot_hist(result, torsion_show="all", n_cluster=10, kind="free_energy")
show_compound(result["molecule"], result["quads"], torsion_enumerate=True)
```

1. **Round 1:** HDBSCAN on every torsion.
2. **Round 2:** a torsion on which no pair of round-1 clusters differs is dropped and HDBSCAN is fitted
   again on the rest. "Differs" is a permutation test: the two clusters' frames are pooled and
   reshuffled to the same sizes `torsion_null_test` times (default 40), and the observed circular W2
   distance must exceed the 95th percentile of the reshuffled ones. The percentile of 40 shuffles is
   noisy: a torsion whose observed W2 is close to it can flip with `seed`. A torsion that a symmetry
   operation moves together with a kept torsion is never dropped; without it the operation maps one
   state onto the other (the ring flip carries the aryl-CO and the aryl-N torsion). `result["rounds"]`
   keeps the observed and null W2 of every pair and torsion, what was dropped, and both cluster counts.
3. **Thermodynamic cut.** On the circular free-energy profile of one torsion in a node (72 bins, F in kT,
   an empty bin set to `max(F) + 1`), cut 1 is the centre of the highest plateau and cut 2 is any bin
   whose F exceeds the lowest point of the smaller arc by at least `kT_cut` (default **1.6 kT**, 0.95
   kcal/mol at 300 K; the rule needs no temperature). Both arcs must hold at least `mass_floor` of all
   frames (default 1 %). Two cuts are needed on a circle: a low route through a shallow third well means
   no split. The cut is taken directly only if that torsion is the only one with such a pair, because the
   marginal profile has integrated every other dimension: an empty cut bin means no density on the whole
   hyperplane, so nothing connects the arcs. Frames within one bin of a **non-empty** cut are marked
   ambiguous.
4. **Bits.** For each torsion, clusters are grouped by pairwise W2 (complete linkage, 20 degrees), the
   groups are numbered from 0 by the frames they own, and clusters with identical bit vectors are merged
   (`merge_by_bits=False` keeps the table and merges nothing). The 20 degrees is below the W2 noise floor
   of broad or multimodal profiles, so it is a coarse grouping, not a statistical test.
5. **Numbering** is by cluster mass, largest first. Neighbouring numbers are not similar structures.

Frames in no leaf get `NOISE_LABEL`, frames beside a non-empty cut `AMBIGUOUS_LABEL`. `result["rmsd"]` is
the heavy-atom symmetry-corrected inter- and intra-cluster RMSD in the final numbering.

* **Observed on one trajectory, not a test of the method.** Folic acid dianion, REST2 state 0, 100 ns at
  stride 10 (5000 frames), 14 torsions (nine heavy-atom, five polar-hydrogen): 12 clusters, 53 % of the
  frames unassigned (32 of them ambiguous), three cuts, all on the CH2-C6 torsion, and no bit merge.
  Wells under 1 % of the frames stay inside their cluster: a floor sets what is a cluster. That
  trajectory is not registered, so the numbers cannot be reproduced from this repository.
* **A cluster is a density basin, not a metastable state;** one trajectory gives no resampling stability,
  and with a few dozen frames in a node the histogram rules are noisy.
* **Not done:** frame weights (refused), a cache of the per-torsion distance terms (exact, but 0.86 s to
  build against 0.07 s to recompute at 1000 frames), and detection of sub-structure below the floor.

## Where this is used

* [Paracetamol: torsion distributions in one labelling](../../tutorial/paracetamol/analysis/symmetry-alignment.ipynb)
* [Paracetamol: symmetry-first clustering](../../tutorial/paracetamol/analysis/clustering.ipynb)
* [c(RGDfV): conformations of a cyclic peptide](../../tutorial/rgdfv/analysis/clustering.ipynb)
