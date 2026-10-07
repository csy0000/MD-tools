# Analysis: paracetamol

Four notebooks over the runs the [cMD](../cMD.md) and [REST2](../REST2.md) pages produced.

| notebook | what it shows |
|---|---|
| [`pmf-1d.ipynb`](pmf-1d.ipynb) | the free energy along each of the three torsions |
| [`clustering.ipynb`](clustering.ipynb) | symmetry-first clustering: the symmetry enumerated, the distance built from it, HDBSCAN, the 18-of-20 assignment, populations and representative structures |
| [`clustering-comparison.ipynb`](clustering-comparison.ipynb) | the historical cluster-then-merge workflow, the new vote alone, and symmetry-first, on the same frames |
| [`mutual-information.ipynb`](mutual-information.ipynb) | which torsions move together, and the estimator's own bias |

**Committed with their outputs**, and rendered here as pages: the figures and numbers come from
the runs these tutorials describe. To run them yourself:

```bash
micromamba activate analysis-env          # see Installing, step 5
export MD_TUTORIAL_RUNS=/path/to/your/tutorial/runs
jupyter lab docs/tutorial/paracetamol/analysis/
```

## Why paracetamol is the interesting case

Three torsions, and the REST2 convention treats them differently. The amide **omega** and all six
aromatic ring bonds are protected — unscaled in every state — so what REST2 heats are the bonds
OUTSIDE the ring: **aryl** (the ordinary torsion about the exocyclic C(aryl)–N bond, which is not
part of the ring) and **phenol** (the hydroxyl rotation).

The phenyl ring also has a two-fold axis, which makes this the smallest honest example of a
symmetry-aware distance ([methods](../../../openmm_methods/analysis/README.md#t_hdbscan)):

* the ring flip RENAMES the two sides of the ring, so a frame measured through C5 and the same
  frame measured through C10 are one configuration; the distance is the minimum over those
  namings, and symmetry-related frames are at distance zero before HDBSCAN fits anything;
* the flip is the **coupled** case: it moves `aryl` and `phenol` together, and both of its images
  (1-3-4-10 and 9-7-8-17) fall outside the torsion table, so they are recomputed from coordinates
  -- not approximated by a 180 degree shift, which on this trajectory would be wrong by up to 25
  degrees;
* the fit finds **two** clusters, A at mass 0.514 and B at 0.484, with 0.225 % density noise and
  nothing ambiguous. The four namings of two conformers never become four clusters.

The [previous page](../../archived/0.6.4/paracetamol/clustering.ipynb), archived unchanged, found
four clusters and merged them into the same two groups afterwards; the comparison notebook shows
the two workflows agree on 3998 of 4000 frames and why the other two differ.

## Related

* [Methods: analysis](../../../openmm_methods/analysis/README.md) — the theory behind the tools, and the migration from the old call
* [REST2: paracetamol](../REST2.md) — where the protected bonds are decided
* [ALA analysis](../../ALA/analysis/index.md) — PMFs on a smaller system
