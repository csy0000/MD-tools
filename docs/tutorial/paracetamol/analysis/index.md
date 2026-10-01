# Analysis: paracetamol

Three notebooks over the runs the [cMD](../cMD.md) and [REST2](../REST2.md) pages produced.

| notebook | what it shows |
|---|---|
| [`pmf-1d.ipynb`](pmf-1d.ipynb) | the free energy along each of the three torsions |
| [`clustering.ipynb`](clustering.ipynb) | **1.** clustering **2.** pruning by symmetry **3.** summary |
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

The phenyl ring also has a two-fold axis, which makes this the smallest honest example of the
thing [`t_symmetry`](../../../openmm_methods/analysis/README.md#t_symmetry) exists for:

* clustering finds **four** basins, two torsions each taking two values;
* the ring flip relates them in pairs, and **not all four merge** — one pairing has the two
  torsions together, the other has them opposite, and those are genuinely different relative
  orientations;
* so four clusters prune to **two groups**, A at mass 0.514 and B at 0.484, with the noise mass
  preserved and the total exactly 1.

The flip is also the **coupled** case: it moves `aryl` and `phenol` together, and both of its
transformed quadruplets fall outside the torsion table, so it is unresolved without coordinates
rather than approximated by a 180° shift.

## Related

* [Methods: analysis](../../../openmm_methods/analysis/README.md) — the theory behind the three tools
* [REST2: paracetamol](../REST2.md) — where the protected bonds are decided
* [ALA analysis](../../ALA/analysis/index.md) — PMFs on a smaller system
