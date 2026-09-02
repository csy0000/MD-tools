"""Compatibility facade: `md_tools.remd` under its pre-v0.5 name.

No logic here. See `md_tools/runtime/__init__.py` for why this exists.
"""
from __future__ import annotations

from ..remd.generated import replica_main, run_generated_remd, run_remd

__all__ = ["replica_main", "run_remd", "run_generated_remd"]
