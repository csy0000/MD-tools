"""Compatibility path for the canonical configuration package.

The code moved to `md_templates.core.config` in Phase 3 of the PR3-PR8 migration, because it is
engine-neutral: it has no OpenMM, OpenFF or RDKit import at any level, and it is what every future
engine will compile its inputs into.

`md_templates.openmm.spec` and every submodule under it remain importable and are the *same module
objects* as their `md_templates.core.config` counterparts -- see `md_templates.openmm._compat` for
why aliasing rather than re-exporting is the safe choice.
"""
from __future__ import annotations

from md_templates.core import config as _config
from md_templates.core.config import (  # noqa: F401
    adapter,
    canonical,
    diffs,
    migrate,
    models,
    resolve,
    units,
)

from .._compat import alias_module, alias_submodules

_SUBMODULES = ("units", "models", "canonical", "resolve", "diffs", "migrate", "adapter")
alias_submodules(__name__, _config, _SUBMODULES)

# The package itself is aliased too, not only its submodules, so `md_templates.openmm.spec` IS
# `md_templates.core.config`. Anything less makes `spec.resolve is core.config.resolve` true while
# `spec is core.config` is false, which is precisely the kind of half-alias that lets a monkeypatch
# land on one object and the code read the other.
alias_module(__name__, _config)

__all__ = list(_SUBMODULES)
