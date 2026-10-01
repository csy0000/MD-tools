"""Analysis of what a run produced. Nothing here integrates anything.

TWO PUBLIC ENTRY POINTS, deliberately independent of each other:

```python
from md_tools.analysis import t_hdbscan, torsional_mi
```

* `t_hdbscan(torsions, ...)` -- a density partition of a torsional ensemble, as a fit/predict
  estimator. Needs scikit-learn, which is the `analysis` extra.
* `torsional_mi(torsions, ...)` -- mutual information between torsion pairs, with the plug-in
  estimator's upward bias measured per pair against a shuffled null. numpy only.

Neither imports the other, and this package imports neither at module load: both are reached
lazily below, so `import md_tools.analysis` costs nothing and a missing extra is reported by
name rather than as a traceback from inside a fit.

THREE THINGS THAT GOVERN EVERY USE OF THE CLUSTERING, and belong here rather than in a function's
docstring because they are not about any one call:

* **A cluster is a density basin, NOT a metastable state.** Nothing in this package looks at time.
  Two density peaks separated by a sampled region are two clusters whatever the barrier between
  them, and a long-lived state that is geometrically diffuse may be one cluster or none.
  Metastability needs dynamical evidence these estimators do not have.
* **Seed agreement is PRECISION, not accuracy.** `draw_agreement_` says the cluster count was
  stable under resampling. It does not say a small state was seen.
* **With non-uniform weights and `resampling=False`, only the POPULATIONS are reweighted.**
  HDBSCAN cannot weight its density estimate -- the core distance is a k-th nearest neighbour
  distance, a rank statistic, so there is no `sample_weight` -- and the partition is therefore
  still fitted on the sampled ensemble. `resampling=N` fits the target distribution itself.

The estimators were written by the hpREST2 project and ported here with that session's agreement;
the reasoning in their docstrings is theirs.
"""
from __future__ import annotations

__all__ = ["THDBSCAN", "t_hdbscan", "torsional_mi"]


def __getattr__(name):
    """Lazy, so neither sklearn nor the metric layer loads until something is actually used.

    THE IMPLEMENTATION MODULES ARE PRIVATE (`_t_hdbscan`, `_torsional_mi`) and that is not a
    style choice. A submodule named `t_hdbscan` beside a callable named `t_hdbscan` is a
    collision Python resolves in the submodule's favour: importing it BINDS it onto the package,
    after which this `__getattr__` is never consulted and `from md_tools.analysis import
    t_hdbscan` silently hands back a module. It is silent because a module has a `__name__` too,
    so a check that prints the name looks right -- which is exactly how it was nearly missed
    here. Calling it then fails with "'module' object is not callable", a long way from the
    import that caused it.
    """
    if name in ("THDBSCAN", "t_hdbscan"):
        from ._t_hdbscan import THDBSCAN, t_hdbscan

        return {"THDBSCAN": THDBSCAN, "t_hdbscan": t_hdbscan}[name]
    if name == "torsional_mi":
        from ._torsional_mi import torsional_mi

        return torsional_mi
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
