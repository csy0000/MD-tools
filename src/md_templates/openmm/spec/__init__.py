"""Compatibility package for the canonical configuration modules.

Most of what lived here moved to `md_templates.core.config` in Phase 3, because it is engine-neutral:
typed models, units, resolution and precedence, canonical serialisation, the hash projections. Each
of those names is bound here to the *same module object* in core, so monkeypatching and shared state
behave exactly as they did -- see `md_templates.openmm._compat`.

`adapter` is the exception and stayed with the engine. It translates the canonical model into this
engine's runtime configuration and reads this engine's `DEFAULTS`; putting it in core made core
import an engine, which the Phase 3 slow gate caught.

That split is also why this package is not itself aliased onto `md_templates.core.config`. If it
were, `core.config` would have to carry `adapter` to keep `from md_templates.openmm.spec import
adapter` working -- which is precisely the dependency the phase removed. A compatibility package
whose members are aliases keeps both promises: the old imports resolve, and core stays clean.
"""
from __future__ import annotations

from md_templates.core import config as _config
from md_templates.core.config import (  # noqa: F401
    canonical,
    diffs,
    migrate,
    models,
    resolve,
    units,
)

from .. import adapter  # noqa: F401
from .._compat import alias_module, alias_submodules

_CORE_SUBMODULES = ("units", "models", "canonical", "resolve", "diffs", "migrate")
alias_submodules(__name__, _config, _CORE_SUBMODULES)
alias_module(f"{__name__}.adapter", adapter)

__all__ = list(_CORE_SUBMODULES) + ["adapter"]
