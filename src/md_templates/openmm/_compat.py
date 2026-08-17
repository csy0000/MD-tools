"""Aliasing helper for the legacy `md_templates.openmm.*` import paths.

Phase 3 of the migration moved the engine-neutral modules into `md_templates.core`. The old import
paths must keep working -- they are a published interface and the campaign explicitly forbids
removing them -- so each one is bound to the module that now holds the code.

**Aliased, not re-exported.** `sys.modules["md_templates.openmm.runstate"]` becomes the *same object*
as `md_templates.core.persistence`, rather than a new module that copies names out of it. The
difference matters in three ways a `from x import *` shim would get wrong:

* module-level state is shared, so a caller that sets `md_templates.openmm.runstate.SOMETHING`
  affects the real module rather than a private copy that nothing reads;
* `monkeypatch.setattr("md_templates.openmm.runstate.f", ...)` patches the function the code actually
  calls -- a copying shim silently patches a shadow and the test passes while testing nothing;
* private names (`_helper`) stay reachable, and existing tests do reach for them.

The cost is that `__module__` on the moved objects now reports the new location. That is a real,
visible change, recorded in the campaign journal rather than hidden: the *names* remain importable
from the old paths, which is the compatibility contract, while where they live is the architecture
change the migration exists to make.
"""
from __future__ import annotations

import sys
from types import ModuleType


def alias_module(legacy_name: str, target: ModuleType) -> ModuleType:
    """Bind `legacy_name` to `target` in `sys.modules` and return it."""
    sys.modules[legacy_name] = target
    return target


def alias_submodules(legacy_package: str, target_package: ModuleType, names) -> None:
    """Bind `legacy_package.<name>` to the matching attribute of `target_package`."""
    for name in names:
        alias_module(f"{legacy_package}.{name}", getattr(target_package, name))
