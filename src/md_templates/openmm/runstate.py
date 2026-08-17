"""Compatibility path for `md_templates.core.persistence`.

Moved in Phase 3 of the PR3-PR8 migration: this module is engine-neutral and now lives in core.
The name here is bound to the same module object, so shared state, monkeypatching and private
names all continue to work -- see `md_templates.openmm._compat`.
"""
from __future__ import annotations

from md_templates.core import persistence as _target

from ._compat import alias_module

alias_module(__name__, _target)

# `save_restart`/`load_restart` need a running Simulation, so Phase 3 left them with the engine.
# Bound onto the aliased core module so the historical `runstate.save_restart(...)` keeps working
# from this import path, which is what every existing caller and test uses.
from .restart import load_restart, save_restart  # noqa: E402

_target.save_restart = save_restart
_target.load_restart = load_restart
