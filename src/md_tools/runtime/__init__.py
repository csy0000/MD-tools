"""Compatibility only. Kept so a project generated before v0.5 still runs.

A pre-v0.5 generated script begins:

    from md_tools.runtime.stage import stage_main
    from md_tools.runtime.replica import replica_main
    from md_tools.runtime.ais import ais_main

Those modules moved -- to `md_tools.md`, `md_tools.remd` and `md_tools.ais` -- when the runtime
stopped pretending to be a directory of templates. The names here re-export the new implementation
and contain **no scientific logic of their own**, so there is no second authority to drift: a
correction lands in one place and both entry points see it.

New work must not import from here. `docs/openmm_methods/README.md` documents the current API.
"""
from __future__ import annotations

__all__ = ["ais", "replica", "stage"]
