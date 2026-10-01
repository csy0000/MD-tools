# Analysis: alchemical free energy

One notebook over a prepared leg from the [hydration free energy](../hydration.md) page.

| notebook | what it shows |
|---|---|
| [`free-energy.ipynb`](free-energy.ipynb) | MBAR, BAR, TI and both EXP directions over one set of samples |

```bash
micromamba activate analysis-env          # see Installing, step 5
export MD_ALCHEMY_LEG=/path/to/AHFE-leg1-run1
jupyter lab docs/tutorial/ethanol/analysis/
```

## Four estimators are not four measurements

They are four probes of ONE dataset. Their spread says how well the ladder overlaps, not how
uncertain the free energy is — **the uncertainty comes from repeats.** If the estimators agree,
the sampling supports all of them; if they disagree, read the overlap before the physics, because
EXP in one direction is the first to break when its exponential average is carried by its tail.
MBAR far from BAR would mean something is wrong with the samples themselves.

The run in the notebook gives MBAR **−3.787 ± 0.103 kcal/mol** with a smallest neighbour overlap
of 0.178 — a ladder is in trouble below about 0.03. That quoted uncertainty is MBAR's asymptotic
covariance for one ladder, and it is **not** the error bar to publish.

## Related

* [Methods: alchemical topology](../../../openmm_methods/alchemy/README.md)
* [What the runtime computes](../../../openmm_methods/alchemy/validation.md)
