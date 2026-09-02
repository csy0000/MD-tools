"""Compatibility facade: `md_tools.ais` under its pre-v0.5 name.

No logic here. See `md_tools/runtime/__init__.py` for why this exists.
"""
from __future__ import annotations

from ..ais.run import ais_main, run_generated_ais

__all__ = ["ais_main", "run_generated_ais"]
