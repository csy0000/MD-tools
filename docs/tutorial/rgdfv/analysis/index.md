# Analysis: c(RGDfV)

One notebook, over the **unscaled state** of an explicit-solvent REST2 ladder that is already
registered:

| notebook | what it shows |
|---|---|
| [`clustering.ipynb`](clustering.ipynb) | `t_hdbscan` with the 1 % floor on 17 torsions, the backbone table, representative structures, how the choice of torsions changes the answer (with the Asp and Arg side chains folded by canonicalization), the clustering on all 24 heavy-atom torsions, the symmetry-corrected heavy-atom RMSD between clusters, and the torsion partitioning of the backbone on all 300 000 frames, its splitting tree, which run each cluster comes from, and the two methods compared |

The methods are in [Methods: torsional clustering](../../../openmm_methods/analysis/torsional-clustering/torsion-preprocessing.ipynb)
([HDBSCAN](../../../openmm_methods/analysis/torsional-clustering/hdbscan.ipynb), [torsion partitioning](../../../openmm_methods/analysis/torsional-clustering/torsion-partitioning.ipynb)); the notebook shows results.

## The data

Registered dataset `hpREST2/RGDfV-explicit-HMR-REST2` (`$MD_DATA/2026/hpREST2/RGDfV-explicit-HMR-REST2`):
c(RGDfV), 79 solute atoms, explicit TIP3P, an 8-state ladder with tau up to 0.5, 500 ns per replica,
three independent runs. **State 0 is the unscaled Hamiltonian**, so `solute_state0_prod1.nc` of each
run is the physical ensemble; the other seven states are hot and are not used. HDBSCAN reads every
75th frame of each run (375 ps, 1334 frames per run, 4002 in all); the torsion partitioning reads every frame.

The dataset was run with md-tools `0.5.4` and registered afterwards; this analysis ran on md-tools `0.6.5.2` with `plot_split_tree`, which is not in a release yet.

To run it yourself:

```bash
micromamba activate analysis-env          # see Installing, step 5
export MD_DATA=/path/to/your/data-root
jupyter lab docs/tutorial/rgdfv/analysis/
```

## What is not here

The notebook does not run a simulation and does not compare the REST2 ladder with a plain run, and
the clusters it finds are **density basins, or regions bounded by barriers, not metastable states**:
nothing in it looks at time.
The 1 % floor, the torsion set and `allow_single_cluster` each change the answer, and the last section
of the notebook shows by how much.

## Related

* [Ligand parameterization](../../ligand-parameterization/index.md) — where c(RGDfV) is built as a peptide-like molecule
* [Paracetamol analysis](../../paracetamol/analysis/index.md) — the same workflow on a molecule with one ring flip
