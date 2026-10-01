# Analysis: alanine dipeptide

Two notebooks over the runs the [cMD](../cMD.md) and [REST2](../REST2.md) pages produced.

| notebook | what it shows |
|---|---|
| [`pmf-1d.ipynb`](pmf-1d.ipynb) | the free energy along phi, cMD against the REST2 cold rung |
| [`pmf-2d.ipynb`](pmf-2d.ipynb) | the Ramachandran surface the 1D projection averages away |

**They are committed with their outputs**, and rendered here as pages: every figure and number
below comes from the runs these tutorials describe, not from a build-time re-execution. To run
them yourself:

```bash
micromamba activate analysis-env          # see Installing, step 5
export MD_TUTORIAL_RUNS=/path/to/your/tutorial/runs
jupyter lab docs/tutorial/ALA/analysis/
```

The root is not defaulted, deliberately: a default would be one machine's path. A notebook that
cannot find it says so by name, with the layout it expects.

## What a PMF here is, and is not

`F(x) = -kT ln p(x)` over a sampled histogram. A free energy along ONE coordinate with everything
else averaged over — not an energy barrier, and not a rate.

**Empty bins stay empty.** A bin nothing visited has no estimate, and filling it in would draw a
barrier whose height is set by the sample size rather than by the physics. They are left as NaN.

**Neither notebook claims convergence.** Showing that a PMF has converged needs repeats or a block
analysis, which is a different question from how to compute one.

## Related

* [Methods: analysis](../../../openmm_methods/analysis/README.md) — the tools unique to md-tools
* [Paracetamol analysis](../../paracetamol/analysis/index.md) — clustering, symmetry, MI
