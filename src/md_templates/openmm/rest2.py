"""Compatibility path for `md_templates.engines.openmm.methods.rest2`.

The OpenMM implementation moved under `md_templates.engines.openmm` in Phase 4, with the methods in
their own subpackage. This name is bound to the same module object, so shared state, monkeypatching
and private names all behave as before -- see `md_templates.openmm._compat`.
"""
from __future__ import annotations

import md_templates.engines.openmm.methods.rest2 as _target

from ._compat import alias_module

alias_module(__name__, _target)
