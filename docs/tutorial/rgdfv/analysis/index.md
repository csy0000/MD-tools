# Analysis: c(RGDfV)

One notebook, over the **unscaled state** of an explicit-solvent REST2 ladder that is already
registered:

| notebook | what it shows |
|---|---|
| [`clustering.ipynb`](clustering.ipynb) | the symmetry enumerated (four operations on the torsions, sixteen once the carboxylate and the guanidinium are included), the columns and their weights, `t_hdbscan` with the 1 % floor, every frame written in the labelling nearest the density peak, the backbone table, representative structures, how the choice of torsions changes the answer, and the resonance-equivalent atoms of the Asp and Arg side chains |

The explanation is in
[Methods: symmetry-aware clustering and the torsion representation](../../../openmm_methods/analysis/clustering-and-symmetry.md);
the notebook shows results only.

## The data

Registered dataset `hpREST2/RGDfV-explicit-HMR-REST2` (`$MD_DATA/2026/hpREST2/RGDfV-explicit-HMR-REST2`):
c(RGDfV), 79 solute atoms, explicit TIP3P, an 8-state ladder with tau up to 0.5, 500 ns per replica,
three independent runs. **State 0 is the unscaled Hamiltonian**, so `solute_state0_prod1.nc` of each
run is the physical ensemble; the other seven states are hot and are not used. The notebook reads
every 75th frame of each run (375 ps, 1334 frames per run, 4002 in all).

The dataset was run with md-tools `0.5.4` and registered afterwards; this analysis ran on `0.6.5` with
the three methods that are not released yet (`THDBSCAN.density_peak`, `THDBSCAN.align_all`,
`SymmetricTorsionDescriptor.align_to_reference`, and the `resonance` option), and the notebook says so at the top.

To run it yourself:

```bash
micromamba activate analysis-env          # see Installing, step 5
export MD_DATA=/path/to/your/data-root
jupyter lab docs/tutorial/rgdfv/analysis/
```

## What is not here

The notebook does not run a simulation and does not compare the REST2 ladder with a plain run, and
the two clusters it finds are **density basins, not metastable states**: nothing in it looks at time.
The 1 % floor, the torsion set and `allow_single_cluster` each change the answer, and the last section
of the notebook shows by how much.

## Related

* [Ligand parameterization](../../ligand-parameterization/index.md) — where c(RGDfV) is built as a peptide-like molecule
* [Paracetamol analysis](../../paracetamol/analysis/index.md) — the same workflow on a molecule with one ring flip
