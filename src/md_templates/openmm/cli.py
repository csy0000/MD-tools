"""Compatibility path for `md_templates.engines.openmm.cli`.

The implementation moved under `md_templates.engines.openmm` in Phase 4. This module keeps both
historical entry points working:

* `from md_templates.openmm.cli import main` and the installed `md-openmm` script, via the module
  alias, so callers get the real module object rather than a copy;
* `python -m md_templates.openmm.cli`, which the alias alone cannot provide -- running a module as
  `__main__` executes THIS file, and without an explicit guard it would import, alias, and exit
  silently with status 0. That is exactly what happened when the alias was first written, and the
  outside-the-checkout test caught it: a command that printed nothing and reported success.

When executed as `__main__` the alias is deliberately skipped: rebinding `sys.modules["__main__"]`
would replace the running program's own module.
"""
from __future__ import annotations

from md_templates.engines.openmm import cli as _target

from ._compat import alias_module

main = _target.main
build_parser = _target.build_parser

if __name__ != "__main__":
    alias_module(__name__, _target)
else:  # pragma: no cover - exercised by `python -m md_templates.openmm.cli`
    raise SystemExit(main())
