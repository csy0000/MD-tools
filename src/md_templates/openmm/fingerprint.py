"""Compatibility path for `md_templates.core.fingerprint`.

Moved in Phase 3 of the PR3-PR8 migration: this module is engine-neutral and now lives in core.
The name here is bound to the same module object, so shared state, monkeypatching and private
names all continue to work -- see `md_templates.openmm._compat`.
"""
from __future__ import annotations

from md_templates.core import fingerprint as _target

from ._compat import alias_module

alias_module(__name__, _target)
