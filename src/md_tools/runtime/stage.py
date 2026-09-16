"""Compatibility facade: `md_tools.md.stage` under its pre-v0.5 name.

No logic here. See `md_tools/runtime/__init__.py` for why this exists.
"""
from __future__ import annotations

from ..md.stage import (EXTENDABLE_FIELDS, check_timestep_against_masses, run_generated_stage,
                        run_stage, solute_atom_indices, stage_main, stage_parser)

__all__ = ["stage_main", "stage_parser", "run_stage", "run_generated_stage",
           "check_timestep_against_masses", "solute_atom_indices", "EXTENDABLE_FIELDS"]
