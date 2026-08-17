"""Compatibility path for `md_templates.core.bundle`.

Moved in Phase 3 of the PR3-PR8 migration: this module is engine-neutral and now lives in core.
The name here is bound to the same module object, so shared state, monkeypatching and private
names all continue to work -- see `md_templates.openmm._compat`.
"""
from __future__ import annotations

from md_templates.core import bundle as _target

from ._compat import alias_module

alias_module(__name__, _target)

# `topology_counts` and `forcefield_provenance` need a built OpenMM system, so Phase 3 left them
# with the engine. Bound onto the aliased core module so the historical
# `bundlev2.topology_counts(...)` keeps working from this import path.
from .bundleinfo import forcefield_provenance, topology_counts  # noqa: E402

_target.topology_counts = topology_counts
_target.forcefield_provenance = forcefield_provenance
