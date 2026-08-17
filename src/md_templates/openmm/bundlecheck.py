"""Compatibility path for `md_templates.engines.openmm.bundlecheck`.

The OpenMM implementation moved under `md_templates.engines.openmm` in Phase 4 so that one engine
provider serves every method and template. This name is bound to the same module object, so shared
state, monkeypatching and private names all behave as before -- see `md_templates.openmm._compat`.
"""
from __future__ import annotations

from md_templates.engines.openmm import bundlecheck as _target

from ._compat import alias_module

alias_module(__name__, _target)
