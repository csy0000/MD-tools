"""Analysis of what a run produced. Nothing here integrates anything.

TWO PUBLIC ENTRY POINTS, deliberately independent of each other:

```python
from md_tools.analysis import t_hdbscan, torsional_mi
```

* `t_hdbscan(torsions, ...)` -- a density PARTITION of a torsional ensemble, as a fit/predict
  estimator. Needs scikit-learn.
* `t_mi(torsions, ...)` -- mutual information between torsion pairs (alias `torsional_mi`), with
  the plug-in estimator's upward bias measured per pair against a shuffled null; plus
  `mi_matrix` and `dependence_graph` for CORRELATION DETECTION across the whole set, with a
  false-discovery-rate correction, because d torsions means d(d-1)/2 simultaneous comparisons.
  numpy only.
* `t_symmetry` / `TorsionSymmetry` -- which of those clusters are the SAME conformer seen through
  a molecular symmetry, decided from RDKit graph automorphisms and a joint-distribution
  comparison, never from equal populations. Needs rdkit.

None of them imports another's estimator, and this package imports none of them at module load:
all are reached lazily below, so `import md_tools.analysis` costs nothing and a missing extra is reported by
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

__all__ = ["THDBSCAN", "t_hdbscan",
           "t_mi", "torsional_mi", "mi_matrix", "dependence_graph",
           "t_symmetry", "TorsionSymmetry", "load_torsion_json", "load_cluster_torsion",
           "draw_torsions", "verify_atom_mapping", "calibrate", "ComparisonTolerance",
           "group_name", "representative_by_vote", "strip_nonpolar_hydrogens",
           "draw_representative_structures"]

_LAZY = {
    "THDBSCAN": "._t_hdbscan", "t_hdbscan": "._t_hdbscan",
    "t_mi": "._t_mi", "torsional_mi": "._t_mi",
    "mi_matrix": "._t_mi", "dependence_graph": "._t_mi",
    "t_symmetry": "._t_symmetry", "TorsionSymmetry": "._t_symmetry",
    "load_torsion_json": "._t_symmetry", "load_cluster_torsion": "._t_symmetry",
    "draw_torsions": "._t_symmetry", "verify_atom_mapping": "._t_symmetry",
    "calibrate": "._t_symmetry", "ComparisonTolerance": "._t_symmetry",
    "group_name": "._t_symmetry", "representative_by_vote": "._t_symmetry",
    "strip_nonpolar_hydrogens": "._t_symmetry",
    "draw_representative_structures": "._t_symmetry",
}


def __getattr__(name):
    """Lazy, so neither sklearn nor the metric layer loads until something is actually used.

    THE IMPLEMENTATION MODULES ARE PRIVATE (`_t_hdbscan`, `_t_mi`, `_t_symmetry`) and that is not a
    style choice. A submodule named `t_hdbscan` beside a callable named `t_hdbscan` is a
    collision Python resolves in the submodule's favour: importing it BINDS it onto the package,
    after which this `__getattr__` is never consulted and `from md_tools.analysis import
    t_hdbscan` silently hands back a module. It is silent because a module has a `__name__` too,
    so a check that prints the name looks right -- which is exactly how it was nearly missed
    here. Calling it then fails with "'module' object is not callable", a long way from the
    import that caused it.
    """
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name, __name__), name)


def __dir__():
    return sorted(__all__)
